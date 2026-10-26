"""SPS 解析与 FFmpeg trace_headers 逐字段对照。

对每段录像（--video）：vidlo 从 codec_config 读出的 SPS，每个语法元素的（位位置，名字，值）必须与 trace_headers 打印的
第一个 SPS（即 extradata 里的那个）的前同样多项完全相同；vidlo 读到 default_display_window 为止，trace_headers
后面多出的时序、HRD 等字段不比。另记下解析出的颜色标记与色度位置。

编码器生成的码流用不到一些语法分支（缩放列表数据、参考图像集的集合间预测、PCM、长期参考、子层档次、扩展宽高比等），
--constructed-from 给一段带多个时间子层的录像时，借用它的 VPS，按 H.265 7.3.2.2 手工写一个用到全部这些分支的 SPS，
存成只含 VPS、SPS 的裸 HEVC 码流交给 trace_headers 解析，再同样逐字段对照。结果写进 report.json，输出目录必须不存在。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import vidlo
from vidlo import _hevc

LINE = re.compile(r"^\[trace_headers[^\]]*\]\s+(\d+)\s+(\S+)\s+[01]+ = (-?\d+)$")


def trace_first_sps(input_args: list[str]) -> list[tuple[int, str, int]]:
    result = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", *input_args, "-map", "0:v:0", "-frames:v", "1", "-c", "copy",
                             "-bsf:v", "trace_headers", "-f", "null", "-"], capture_output=True, text=True)
    fields: list[tuple[int, str, int]] = []
    inside = False
    for line in result.stderr.splitlines():
        if not line.startswith("[trace_headers"):
            continue
        if line.rstrip().endswith("Sequence Parameter Set"):
            if fields:
                break
            inside = True
            continue
        if inside:
            if re.search(r"\] [A-Z][A-Za-z ]+$", line):  # 下一个结构的标题
                break
            m = LINE.match(line)
            if m is None:
                raise RuntimeError(f"unexpected trace_headers line: {line}")
            fields.append((int(m[1]), m[2], int(m[3])))
    if not fields:
        raise RuntimeError(f"trace_headers printed no SPS (exit {result.returncode}): {result.stderr[-1500:]}")
    return fields


def compare(sps: _hevc.Sps, trace: list[tuple[int, str, int]]) -> dict[str, Any]:
    ours = sps.fields
    mismatch = next((i for i, (a, b) in enumerate(zip(ours, trace)) if a != b), None)
    result: dict[str, Any] = {
        "passed": len(ours) <= len(trace) and mismatch is None, "fields_compared": len(ours), "trace_fields": len(trace),
        "display_area": sps.display_area, "video_signal_type_present": sps.video_signal_type_present,
        "video_full_range_flag": sps.video_full_range_flag, "matrix_coefficients": sps.matrix_coefficients,
        "colour_primaries": sps.colour_primaries, "transfer_characteristics": sps.transfer_characteristics,
        "chroma_loc_info_present": sps.chroma_loc_info_present,
        "chroma_sample_loc_type": [sps.chroma_sample_loc_type_top_field, sps.chroma_sample_loc_type_bottom_field],
    }
    if mismatch is not None:
        result["first_mismatch"] = {"vidlo": ours[mismatch], "trace_headers": trace[mismatch]}
    elif len(ours) > len(trace):
        result["first_mismatch"] = {"vidlo": ours[len(trace)], "trace_headers": None}
    return result


def check(path: Path) -> dict[str, Any]:
    with vidlo.VideoReader(path, format="nv12") as video:
        config = video.metadata.codec_config
    return compare(_hevc.stream_sps(config), trace_first_sps(["-i", str(path)]))


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def u(self, n: int, value: int) -> None:
        if not 0 <= value < (1 << n):
            raise ValueError(f"{value} does not fit in {n} bits")
        self.bits.extend((value >> (n - 1 - i)) & 1 for i in range(n))

    def ue(self, value: int) -> None:
        code = value + 1
        self.u(code.bit_length() - 1, 0)
        self.u(code.bit_length(), code)

    def se(self, value: int) -> None:
        self.ue(2 * value - 1 if value > 0 else -2 * value)

    def nal(self) -> bytes:
        """加上 rbsp_trailing_bits，再插入防竞争字节。"""
        bits = self.bits + [1] + [0] * ((8 - (len(self.bits) + 1) % 8) % 8)
        rbsp = bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))
        out = bytearray()
        zeros = 0
        for byte in rbsp:
            if zeros >= 2 and byte <= 3:
                out.append(3)
                zeros = 0
            out.append(byte)
            zeros = zeros + 1 if byte == 0 else 0
        return bytes(out)


def profile(w: BitWriter, idc: int, compatible: tuple[int, ...]) -> None:
    """profile_tier_level 里一个档次的 88 位；按 idc 与兼容标志走对应分支，保留位与可选标志都写 0。"""
    w.u(2, 0)
    w.u(1, 0)
    w.u(5, idc)
    for j in range(32):
        w.u(1, 1 if j in compatible else 0)
    w.u(4, 0b1001)  # progressive、interlaced、non_packed、frame_only
    w.u(43, 0)
    w.u(1, 0)


def constructed_sps(vps_id: int, max_sub_layers_minus1: int) -> bytes:
    if max_sub_layers_minus1 < 2:
        raise ValueError("the VPS must allow at least 3 temporal sub-layers")
    w = BitWriter()
    w.u(1, 0)
    w.u(6, _hevc.NAL_SPS)
    w.u(6, 0)
    w.u(3, 1)
    w.u(4, vps_id)
    w.u(3, max_sub_layers_minus1)
    w.u(1, 1)
    # 总档次 Main 且兼容 Main 10（走 one_picture_only 分支）；子层 0 只标 Main（43 位保留位分支）、带级别；
    # 子层 1 为 idc 8（约束标志分支、34 位保留位、reserved_zero_bit），不带级别；其余子层都不带
    profile(w, 1, (1, 2))
    w.u(8, 93)
    present = [(1, 1), (1, 0)] + [(0, 0)] * (max_sub_layers_minus1 - 2)
    for profile_present, level_present in present:
        w.u(1, profile_present)
        w.u(1, level_present)
    for _ in range(max_sub_layers_minus1, 8):
        w.u(2, 0)
    profile(w, 1, (1,))
    w.u(8, 90)
    profile(w, 8, (8,))
    w.ue(0)  # sps_seq_parameter_set_id
    w.ue(1)  # chroma_format_idc
    w.ue(64)
    w.ue(64)
    w.u(1, 1)  # conformance_window_flag
    for offset in (1, 2, 0, 3):
        w.ue(offset)
    w.ue(0)
    w.ue(0)
    w.ue(4)  # log2_max_pic_order_cnt_lsb_minus4：长期参考的 lsb 为 8 位
    w.u(1, 1)
    for _ in range(max_sub_layers_minus1 + 1):
        w.ue(4)
        w.ue(2)
        w.ue(0)
    for value in (0, 3, 0, 3, 1, 1):  # 8×8 到 64×64 的编码块，4×4 到 32×32 的变换块
        w.ue(value)
    w.u(1, 1)  # scaling_list_enabled_flag
    w.u(1, 1)  # sps_scaling_list_data_present_flag
    for size_id in range(4):
        for matrix_id in range(0, 6, 3 if size_id == 3 else 1):
            explicit = (size_id + matrix_id) % 2 == 0 or matrix_id == 0
            w.u(1, 1 if explicit else 0)
            if not explicit:
                w.ue(1 if size_id < 3 else 0)
            else:
                if size_id > 1:
                    w.se(5 - matrix_id)
                for i in range(min(64, 1 << (4 + (size_id << 1)))):
                    w.se((i * 7 + matrix_id) % 11 - 5)
    w.u(1, 1)  # amp
    w.u(1, 1)  # sao
    w.u(1, 1)  # pcm_enabled_flag
    w.u(4, 7)
    w.u(4, 6)
    w.ue(0)
    w.ue(1)
    w.u(1, 1)
    w.ue(3)  # num_short_term_ref_pic_sets
    # 集合 0 直接给出：S0 = [-1, -3]，S1 = [3]
    w.ue(2)
    w.ue(1)
    for delta_minus1, used in ((0, 1), (1, 0)):
        w.ue(delta_minus1)
        w.u(1, used)
    w.ue(2)
    w.u(1, 1)
    # 集合 1 由集合 0 预测（deltaRps = -1，参考集合有 3 项，所以 4 组标志）；集合 2 由集合 1 预测（deltaRps = +2，集合 1 推出 3 项）
    for sign, abs_minus1, used, use_delta in ((1, 0, (1, 0, 1, 0), (None, 1, None, 0)), (0, 1, (0, 1, 0, 1), (1, None, 0, None))):
        w.u(1, 1)
        w.u(1, sign)
        w.ue(abs_minus1)
        for flag, delta in zip(used, use_delta):
            w.u(1, flag)
            if not flag:
                w.u(1, delta)
    w.u(1, 1)  # long_term_ref_pics_present_flag
    w.ue(2)
    for lsb, used in ((5, 1), (17, 0)):
        w.u(8, lsb)
        w.u(1, used)
    w.u(1, 1)
    w.u(1, 1)
    w.u(1, 1)  # vui_parameters_present_flag
    w.u(1, 1)
    w.u(8, 255)  # EXTENDED_SAR
    w.u(16, 4)
    w.u(16, 3)
    w.u(1, 1)
    w.u(1, 1)
    w.u(1, 1)  # video_signal_type_present_flag
    w.u(3, 2)
    w.u(1, 1)
    w.u(1, 1)
    w.u(8, 9)
    w.u(8, 14)
    w.u(8, 9)
    w.u(1, 1)  # chroma_loc_info_present_flag
    w.ue(3)
    w.ue(4)
    w.u(3, 0)  # neutral_chroma_indication、field_seq、frame_field_info
    w.u(1, 1)  # default_display_window_flag
    for offset in (1, 2, 3, 4):
        w.ue(offset)
    w.u(1, 0)  # vui_timing_info_present_flag
    w.u(1, 0)  # bitstream_restriction_flag
    w.u(1, 0)  # sps_extension_present_flag
    return w.nal()


def check_constructed(clip: Path, output: Path) -> dict[str, Any]:
    with vidlo.VideoReader(clip, format="nv12") as video:
        config = video.metadata.codec_config
    vps = next(nal for nal_type, nal in _hevc.parse_hvcc(config).nal_units if nal_type == _hevc.NAL_VPS)
    vps_rbsp = _hevc.rbsp(vps)
    vps_id = vps_rbsp[2] >> 4
    max_sub_layers_minus1 = (vps_rbsp[3] >> 1) & 7  # vps_base_layer 两个标志、vps_max_layers_minus1（6 位）之后
    sps = constructed_sps(vps_id, max_sub_layers_minus1)
    path = output / "constructed_sps.hevc"
    path.write_bytes(b"\x00\x00\x00\x01" + vps + b"\x00\x00\x00\x01" + sps)
    result = compare(_hevc.parse_sps(sps), trace_first_sps(["-f", "hevc", "-i", str(path)]))
    names = {name for _, name, _ in _hevc.parse_sps(sps).fields}
    wanted = ("sub_layer_profile_idc[1]", "sub_layer_level_idc[0]", "sub_layer_max_12bit_constraint_flag[1]", "scaling_list_dc_coef_minus8[1][0]",
              "scaling_list_pred_matrix_id_delta[0][1]", "pcm_loop_filter_disabled_flag", "use_delta_flag[1]", "lt_ref_pic_poc_lsb_sps[1]",
              "sar_height", "overscan_appropriate_flag", "def_disp_win_bottom_offset")
    result["branches"] = {name: name in names for name in wanted}
    result["passed"] = result["passed"] and all(result["branches"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--constructed-from", type=Path, help="借用这段录像的 VPS（要有至少 3 个时间子层）构造 SPS")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = {str(path): check(path) for path in args.video}
    if args.constructed_from is not None:
        checks["constructed"] = check_constructed(args.constructed_from, args.output)
    report = {"vidlo_version": vidlo.__version__, "checks": checks, "passed": bool(checks) and all(c["passed"] for c in checks.values())}
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for name, c in checks.items():
        print(("PASS " if c["passed"] else "FAIL ") + name + f"  fields={c['fields_compared']}"
              + ("" if c["passed"] else "  " + json.dumps({k: c.get(k) for k in ("first_mismatch", "branches")})))
    print("ALL PASSED" if report["passed"] else "SOME FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
