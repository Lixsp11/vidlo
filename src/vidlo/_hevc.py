"""HEVC 参数集的只读解析：从 codec_config（hvcC）取出 SPS，读到 VUI 里的颜色标记和色度位置。

NVDEC 的 sequence 回调给出尺寸、位深、颜色范围和矩阵，但不给色度位置；这里从 SPS 补齐，打开录像时就能检查
输出 RGB 的条件。按 H.265（V11，2026-01）7.3.2.2 和附录 E.2.1 逐个语法元素读，名字、顺序与 FFmpeg
cbs_h265（trace_headers 的输出）相同，便于逐字段对照；读到 default_display_window 为止，后面的时序、HRD
等字段用不到。只处理 nuh_layer_id 为 0 的 SPS。
"""

from __future__ import annotations

from dataclasses import dataclass

from ._native import Error, UnsupportedError

NAL_VPS, NAL_SPS, NAL_PPS = 32, 33, 34


@dataclass(frozen=True)
class HvcC:
    nal_length_size: int
    nal_units: list[tuple[int, bytes]]  # (nal_unit_type, 含 2 字节头的 NAL)，按 hvcC 里的顺序


def parse_hvcc(config: bytes) -> HvcC:
    """ISO/IEC 14496-15 的 HEVCDecoderConfigurationRecord：23 字节的固定部分，后面是各类 NAL 的数组。"""
    if len(config) < 23 or config[0] != 1:
        raise UnsupportedError("HEVC codec config is not an hvcC record; not supported yet")
    nal_length_size = (config[21] & 3) + 1
    if nal_length_size == 3:
        raise Error("hvcC NAL length size of 3 bytes is invalid")
    units: list[tuple[int, bytes]] = []
    offset = 23
    for _ in range(config[22]):
        if offset + 3 > len(config):
            raise Error("hvcC array header runs past the end of codec config")
        array_type = config[offset] & 0x3F
        count = int.from_bytes(config[offset + 1:offset + 3], "big")
        offset += 3
        for _ in range(count):
            if offset + 2 > len(config):
                raise Error("hvcC NAL length runs past the end of codec config")
            length = int.from_bytes(config[offset:offset + 2], "big")
            offset += 2
            nal = config[offset:offset + length]
            if length < 2 or len(nal) != length:
                raise Error("hvcC NAL unit runs past the end of codec config")
            if (nal[0] >> 1) & 0x3F != array_type:
                raise Error(f"hvcC array of NAL type {array_type} holds a NAL of type {(nal[0] >> 1) & 0x3F}")
            units.append((array_type, nal))
            offset += length
    if offset != len(config):
        raise Error(f"hvcC has {len(config) - offset} unexpected trailing bytes")
    return HvcC(nal_length_size, units)


def rbsp(nal: bytes) -> bytes:
    """去掉防竞争字节：NAL 里的 00 00 03 还原成 00 00（H.265 7.4.2）。"""
    out = bytearray()
    zeros = 0
    for byte in nal:
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


class _Reader:
    """按位读 RBSP，并按读的顺序记下每个语法元素的（起始位位置，名字，值）。"""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.fields: list[tuple[int, str, int]] = []

    def _bits(self, n: int) -> int:
        if self.pos + n > len(self.data) * 8:
            raise Error("SPS ends in the middle of a syntax element")
        value = 0
        for _ in range(n):
            value = (value << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return value

    def _record(self, start: int, name: str, value: int) -> int:
        self.fields.append((start, name, value))
        return value

    def u(self, n: int, name: str) -> int:
        start = self.pos
        return self._record(start, name, self._bits(n))

    def ue(self, name: str) -> int:
        start = self.pos
        zeros = 0
        while self._bits(1) == 0:
            zeros += 1
            if zeros > 31:
                raise Error(f"invalid Exp-Golomb code for {name}")
        return self._record(start, name, (1 << zeros) - 1 + self._bits(zeros))

    def se(self, name: str) -> int:
        start = self.pos
        zeros = 0
        while self._bits(1) == 0:
            zeros += 1
            if zeros > 31:
                raise Error(f"invalid Exp-Golomb code for {name}")
        k = (1 << zeros) - 1 + self._bits(zeros)
        return self._record(start, name, (k + 1) // 2 if k & 1 else -(k // 2))


@dataclass(frozen=True)
class Sps:
    """SPS 里与读取有关的值；没有出现的字段按标准的推断值填写。fields 是逐个语法元素的记录，供核对。"""

    chroma_format_idc: int
    bit_depth_luma: int
    bit_depth_chroma: int
    coded_width: int
    coded_height: int
    display_area: tuple[int, int, int, int]  # 按 conformance window 裁出的（左，上，右，下），右、下不含
    video_signal_type_present: bool
    video_full_range_flag: int
    colour_primaries: int
    transfer_characteristics: int
    matrix_coefficients: int
    chroma_loc_info_present: bool
    chroma_sample_loc_type_top_field: int
    chroma_sample_loc_type_bottom_field: int
    field_seq_flag: int
    fields: list[tuple[int, str, int]]


def _profile_tier_level(r: _Reader, max_sub_layers_minus1: int) -> None:
    def profile(prefix: str, suffix: str) -> None:
        r.u(2, f"{prefix}profile_space{suffix}")
        r.u(1, f"{prefix}tier_flag{suffix}")
        idc = r.u(5, f"{prefix}profile_idc{suffix}")
        compatible = [r.u(1, f"{prefix}profile_compatibility_flag{suffix}[{j}]") for j in range(32)]
        for name in ("progressive_source_flag", "interlaced_source_flag", "non_packed_constraint_flag",
                     "frame_only_constraint_flag"):
            r.u(1, f"{prefix}{name}{suffix}")

        def is_(*profiles: int) -> bool:
            return any(idc == p or compatible[p] for p in profiles)

        # 只是给这些位起名，各分支的总位数相同（标准原注释）
        if is_(4, 5, 6, 7, 8, 9, 10, 11):
            for name in ("max_12bit_constraint_flag", "max_10bit_constraint_flag", "max_8bit_constraint_flag",
                         "max_422chroma_constraint_flag", "max_420chroma_constraint_flag",
                         "max_monochrome_constraint_flag", "intra_constraint_flag", "one_picture_only_constraint_flag",
                         "lower_bit_rate_constraint_flag"):
                r.u(1, f"{prefix}{name}{suffix}")
            if is_(5, 9, 10, 11):
                r.u(1, f"{prefix}max_14bit_constraint_flag{suffix}")
                r.u(24, f"{prefix}reserved_zero_33bits")
                r.u(9, f"{prefix}reserved_zero_33bits")
            else:
                r.u(24, f"{prefix}reserved_zero_34bits")
                r.u(10, f"{prefix}reserved_zero_34bits")
        elif is_(2):
            r.u(7, f"{prefix}reserved_zero_7bits")
            r.u(1, f"{prefix}one_picture_only_constraint_flag{suffix}")
            # 名字照 cbs_h265：子层这里也叫 reserved_zero_43bits
            reserved = "general_reserved_zero_35bits" if prefix == "general_" else "sub_layer_reserved_zero_43bits"
            r.u(24, reserved)
            r.u(11, reserved)
        else:
            r.u(24, f"{prefix}reserved_zero_43bits")
            r.u(19, f"{prefix}reserved_zero_43bits")
        if is_(1, 2, 3, 4, 5, 9, 11):
            r.u(1, f"{prefix}inbld_flag{suffix}")
        else:
            r.u(1, f"{prefix}reserved_zero_bit")

    profile("general_", "")
    r.u(8, "general_level_idc")
    present = [(r.u(1, f"sub_layer_profile_present_flag[{i}]"), r.u(1, f"sub_layer_level_present_flag[{i}]"))
               for i in range(max_sub_layers_minus1)]
    if max_sub_layers_minus1 > 0:
        for _ in range(max_sub_layers_minus1, 8):
            r.u(2, "reserved_zero_2bits")
    for i, (profile_present, level_present) in enumerate(present):
        if profile_present:
            profile("sub_layer_", f"[{i}]")
        if level_present:
            r.u(8, f"sub_layer_level_idc[{i}]")


def _scaling_list_data(r: _Reader) -> None:
    for size_id in range(4):
        for matrix_id in range(0, 6, 3 if size_id == 3 else 1):
            if not r.u(1, f"scaling_list_pred_mode_flag[{size_id}][{matrix_id}]"):
                r.ue(f"scaling_list_pred_matrix_id_delta[{size_id}][{matrix_id}]")
            else:
                if size_id > 1:
                    r.se(f"scaling_list_dc_coef_minus8[{size_id - 2}][{matrix_id}]")
                for i in range(min(64, 1 << (4 + (size_id << 1)))):
                    r.se(f"scaling_list_delta_coeff[{size_id}][{matrix_id}][{i}]")


def _st_ref_pic_set(r: _Reader, idx: int, sets: list[tuple[list[int], list[int], list[int], list[int]]]) -> None:
    """读第 idx 个候选短期参考图像集，并按 (7-61)、(7-62) 推出它的 DeltaPocS0/S1，供后面的集合预测时引用。

    sets 里每项为（DeltaPocS0，UsedByCurrPicS0，DeltaPocS1，UsedByCurrPicS1）。SPS 里 idx 小于集合总数，
    所以没有 delta_idx_minus1，参考的总是前一个集合。
    """
    predicted = r.u(1, "inter_ref_pic_set_prediction_flag") if idx != 0 else 0
    if predicted:
        ref_s0, _, ref_s1, _ = sets[idx - 1]
        sign = r.u(1, "delta_rps_sign")
        delta_rps = (1 - 2 * sign) * (r.ue("abs_delta_rps_minus1") + 1)
        num_delta_pocs = len(ref_s0) + len(ref_s1)
        used, use_delta = [], []
        for j in range(num_delta_pocs + 1):
            used.append(r.u(1, f"used_by_curr_pic_flag[{j}]"))
            use_delta.append(1 if used[j] else r.u(1, f"use_delta_flag[{j}]"))
        n_neg = len(ref_s0)
        s0, u0, s1, u1 = [], [], [], []
        for j in range(len(ref_s1) - 1, -1, -1):
            d = ref_s1[j] + delta_rps
            if d < 0 and use_delta[n_neg + j]:
                s0.append(d)
                u0.append(used[n_neg + j])
        if delta_rps < 0 and use_delta[num_delta_pocs]:
            s0.append(delta_rps)
            u0.append(used[num_delta_pocs])
        for j in range(n_neg):
            d = ref_s0[j] + delta_rps
            if d < 0 and use_delta[j]:
                s0.append(d)
                u0.append(used[j])
        for j in range(n_neg - 1, -1, -1):
            d = ref_s0[j] + delta_rps
            if d > 0 and use_delta[j]:
                s1.append(d)
                u1.append(used[j])
        if delta_rps > 0 and use_delta[num_delta_pocs]:
            s1.append(delta_rps)
            u1.append(used[num_delta_pocs])
        for j in range(len(ref_s1)):
            d = ref_s1[j] + delta_rps
            if d > 0 and use_delta[n_neg + j]:
                s1.append(d)
                u1.append(used[n_neg + j])
        if len(s0) + len(s1) > 16:
            raise Error(f"short-term reference picture set {idx} has more than 16 pictures")
        sets.append((s0, u0, s1, u1))
        return
    num_negative = r.ue("num_negative_pics")
    num_positive = r.ue("num_positive_pics")
    if num_negative > 16 or num_negative + num_positive > 16:
        raise Error(f"short-term reference picture set {idx} has more than 16 pictures")
    s0, u0, s1, u1 = [], [], [], []
    poc = 0
    for i in range(num_negative):
        poc -= r.ue(f"delta_poc_s0_minus1[{i}]") + 1
        s0.append(poc)
        u0.append(r.u(1, f"used_by_curr_pic_s0_flag[{i}]"))
    poc = 0
    for i in range(num_positive):
        poc += r.ue(f"delta_poc_s1_minus1[{i}]") + 1
        s1.append(poc)
        u1.append(r.u(1, f"used_by_curr_pic_s1_flag[{i}]"))
    sets.append((s0, u0, s1, u1))


def parse_sps(nal: bytes) -> Sps:
    r = _Reader(rbsp(nal))
    r.u(1, "forbidden_zero_bit")
    if r.u(6, "nal_unit_type") != NAL_SPS:
        raise Error("not an SPS NAL unit")
    if r.u(6, "nuh_layer_id") != 0:
        raise UnsupportedError("SPS with nuh_layer_id > 0 (multi-layer HEVC) is not supported yet")
    r.u(3, "nuh_temporal_id_plus1")
    r.u(4, "sps_video_parameter_set_id")
    max_sub_layers_minus1 = r.u(3, "sps_max_sub_layers_minus1")
    r.u(1, "sps_temporal_id_nesting_flag")
    _profile_tier_level(r, max_sub_layers_minus1)
    r.ue("sps_seq_parameter_set_id")
    chroma_format_idc = r.ue("chroma_format_idc")
    if chroma_format_idc == 3:
        r.u(1, "separate_colour_plane_flag")
    width = r.ue("pic_width_in_luma_samples")
    height = r.ue("pic_height_in_luma_samples")
    window = [0, 0, 0, 0]
    if r.u(1, "conformance_window_flag"):
        window = [r.ue(f"conf_win_{side}_offset") for side in ("left", "right", "top", "bottom")]
    bit_depth_luma = r.ue("bit_depth_luma_minus8") + 8
    bit_depth_chroma = r.ue("bit_depth_chroma_minus8") + 8
    log2_max_poc_lsb = r.ue("log2_max_pic_order_cnt_lsb_minus4") + 4
    ordering_present = r.u(1, "sps_sub_layer_ordering_info_present_flag")
    for i in range(0 if ordering_present else max_sub_layers_minus1, max_sub_layers_minus1 + 1):
        r.ue(f"sps_max_dec_pic_buffering_minus1[{i}]")
        r.ue(f"sps_max_num_reorder_pics[{i}]")
        r.ue(f"sps_max_latency_increase_plus1[{i}]")
    for name in ("log2_min_luma_coding_block_size_minus3", "log2_diff_max_min_luma_coding_block_size",
                 "log2_min_luma_transform_block_size_minus2", "log2_diff_max_min_luma_transform_block_size",
                 "max_transform_hierarchy_depth_inter", "max_transform_hierarchy_depth_intra"):
        r.ue(name)
    if r.u(1, "scaling_list_enabled_flag"):
        if r.u(1, "sps_scaling_list_data_present_flag"):
            _scaling_list_data(r)
    r.u(1, "amp_enabled_flag")
    r.u(1, "sample_adaptive_offset_enabled_flag")
    if r.u(1, "pcm_enabled_flag"):
        r.u(4, "pcm_sample_bit_depth_luma_minus1")
        r.u(4, "pcm_sample_bit_depth_chroma_minus1")
        r.ue("log2_min_pcm_luma_coding_block_size_minus3")
        r.ue("log2_diff_max_min_pcm_luma_coding_block_size")
        r.u(1, "pcm_loop_filter_disabled_flag")
    num_sets = r.ue("num_short_term_ref_pic_sets")
    if num_sets > 64:
        raise Error(f"num_short_term_ref_pic_sets {num_sets} exceeds 64")
    sets: list[tuple[list[int], list[int], list[int], list[int]]] = []
    for i in range(num_sets):
        _st_ref_pic_set(r, i, sets)
    if r.u(1, "long_term_ref_pics_present_flag"):
        for i in range(r.ue("num_long_term_ref_pics_sps")):
            r.u(log2_max_poc_lsb, f"lt_ref_pic_poc_lsb_sps[{i}]")
            r.u(1, f"used_by_curr_pic_lt_sps_flag[{i}]")
    r.u(1, "sps_temporal_mvp_enabled_flag")
    r.u(1, "strong_intra_smoothing_enabled_flag")

    # VUI 不出现时各字段取标准规定的推断值（E.3.1）：video_format 5，limited range，颜色描述 2（未指定），色度位置 0
    signal_present = full_range = chroma_loc_present = field_seq = 0
    primaries = transfer = matrix = 2
    loc_top = loc_bottom = 0
    if r.u(1, "vui_parameters_present_flag"):
        if r.u(1, "aspect_ratio_info_present_flag"):
            if r.u(8, "aspect_ratio_idc") == 255:
                r.u(16, "sar_width")
                r.u(16, "sar_height")
        if r.u(1, "overscan_info_present_flag"):
            r.u(1, "overscan_appropriate_flag")
        signal_present = r.u(1, "video_signal_type_present_flag")
        if signal_present:
            r.u(3, "video_format")
            full_range = r.u(1, "video_full_range_flag")
            if r.u(1, "colour_description_present_flag"):
                primaries = r.u(8, "colour_primaries")
                transfer = r.u(8, "transfer_characteristics")
                matrix = r.u(8, "matrix_coefficients")
        chroma_loc_present = r.u(1, "chroma_loc_info_present_flag")
        if chroma_loc_present:
            loc_top = r.ue("chroma_sample_loc_type_top_field")
            loc_bottom = r.ue("chroma_sample_loc_type_bottom_field")
        r.u(1, "neutral_chroma_indication_flag")
        field_seq = r.u(1, "field_seq_flag")
        r.u(1, "frame_field_info_present_flag")
        if r.u(1, "default_display_window_flag"):
            for side in ("left", "right", "top", "bottom"):
                r.ue(f"def_disp_win_{side}_offset")

    # 6.2：4:2:0 时色度宽高都是亮度的一半，conformance window 的偏移以色度样本为单位
    sub_width = 2 if chroma_format_idc in (1, 2) else 1
    sub_height = 2 if chroma_format_idc == 1 else 1
    area = (sub_width * window[0], sub_height * window[2], width - sub_width * window[1], height - sub_height * window[3])
    if not (0 <= area[0] < area[2] <= width and 0 <= area[1] < area[3] <= height):
        raise Error(f"conformance window {window} does not fit the {width}x{height} picture")
    return Sps(chroma_format_idc, bit_depth_luma, bit_depth_chroma, width, height, area, bool(signal_present),
               full_range, primaries, transfer, matrix, bool(chroma_loc_present), loc_top, loc_bottom, field_seq,
               r.fields)


def stream_sps(config: bytes) -> Sps:
    """codec_config 里唯一的 SPS。按帧号读时只送 codec_config 里的参数集，所以要求 SPS 在这里。"""
    sps = [nal for nal_type, nal in parse_hvcc(config).nal_units if nal_type == NAL_SPS]
    if not sps:
        raise UnsupportedError("codec config has no SPS (parameter sets only inside the stream); not supported yet")
    if len(sps) > 1:
        raise UnsupportedError(f"codec config has {len(sps)} SPS; only one is supported yet")
    return parse_sps(sps[0])
