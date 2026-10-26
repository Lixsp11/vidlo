/* libnvcuvid 的链接用占位库：只在构建时让链接器找到这些符号，运行时加载的是驱动自带的 libnvcuvid.so.1。
 * 导出的函数名与 Video Codec SDK 13.0.37 的 Lib/linux/stubs/x86_64/libnvcuvid.so 相同（nm -D 核对过），
 * SONAME 也一样，所以扩展模块的 NEEDED 写的是 libnvcuvid.so.1。函数体不会被调用。
 * 构建：gcc -shared -fPIC -Wl,-soname,libnvcuvid.so.1 -o libnvcuvid.so nvcuvid_stub.c */

#define STUB(name) \
  int name(void) { return 999; /* CUDA_ERROR_UNKNOWN */ }

STUB(cuvidCreateDecoder)
STUB(cuvidCreateVideoParser)
STUB(cuvidCreateVideoSource)
STUB(cuvidCreateVideoSourceW)
STUB(cuvidCtxLock)
STUB(cuvidCtxLockCreate)
STUB(cuvidCtxLockDestroy)
STUB(cuvidCtxUnlock)
STUB(cuvidDecodePicture)
STUB(cuvidDestroyDecoder)
STUB(cuvidDestroyVideoParser)
STUB(cuvidDestroyVideoSource)
STUB(cuvidGetDecodeStatus)
STUB(cuvidGetDecoderCaps)
STUB(cuvidGetSourceAudioFormat)
STUB(cuvidGetSourceVideoFormat)
STUB(cuvidGetVideoSourceState)
STUB(cuvidMapVideoFrame)
STUB(cuvidMapVideoFrame64)
STUB(cuvidParseVideoData)
STUB(cuvidPrivateOp)
STUB(cuvidReconfigureDecoder)
STUB(cuvidSetVideoSourceState)
STUB(cuvidUnmapVideoFrame)
STUB(cuvidUnmapVideoFrame64)
