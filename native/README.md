# PixelPeeper Native（C++17 / C ABI）

这是 `pixelpeeper/analysis.py` 的原生实现，目标是让 Android、HarmonyOS
以及其他不能运行 Python 的平台直接复用同一套检测算法。

- **无第三方依赖**：不依赖 NumPy、Pillow、OpenCV、Eigen。
- **稳定 C ABI**：`extern "C"`，结果结构体由调用方分配，不需要跨语言释放内存。
- **可重入**：所有状态都在单次调用内，多个线程可用各自的 `pp_result` 并发分析。
- **同一套算法**：重复像素检测、重采样拟合、黑边、多档位候选、2× 系统放大拆层。
- **已回归验证**：与 Python 版 10 个对比场景逐项对齐，方法、倍率、内部分辨率全部一致；
  1520×940 样图耗时约 **0.64 s**（Python 约 5.6 s）。

> 核心只接收解码后的像素。PNG/JPEG/WebP 解码仍由平台完成：
> Android 用 `ImageDecoder`/`Bitmap`，HarmonyOS 用 `image.createImageSource`/`PixelMap`。
> 这是刻意的设计：避免把 libpng/libjpeg 或各平台图形栈绑进算法库。

## 目录

```text
native/
├─ include/pixelpeeper.h              # 公开 C ABI
├─ src/pixelpeeper.cpp                # 检测引擎
├─ tools/pp_cli.cpp                   # 读取 .rgb 原始像素并输出 JSON（回归测试用）
├─ tests/smoke.c                      # 纯 C 调用冒烟测试
├─ tests/test_parity.py               # 与 Python 参考实现的对比测试
├─ platform/android/                  # JNI + Kotlin 封装
├─ platform/harmony/                  # HarmonyOS Node-API 封装
└─ CMakeLists.txt
```

## 编译（Windows / Linux / macOS）

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j
ctest --test-dir native/build --output-on-failure
```

产物：

- `libpixelpeeper.so` / `libpixelpeeper.dylib` / `pixelpeeper.dll`
- `pp_cli`（测试工具）
- `pp_smoke`（C ABI 冒烟测试）

静态库：

```bash
cmake -S native -B native/build-static -DPP_BUILD_SHARED=OFF -DPP_BUILD_CLI=OFF
```

## C 调用

```c
#include "pixelpeeper.h"

pp_result r;
int rc = pp_analyze_rgb(rgb, width, height, width * 3, NULL, 0, &r);
if (rc != 0) {
    fprintf(stderr, "%s\n", r.error);
    return;
}
printf("%s %.3fx -> %dx%d (confidence %.2f)\n",
       r.method == PP_METHOD_NEAREST ? "nearest" :
       r.method == PP_METHOD_SMOOTH  ? "smooth" : "none",
       r.scale, r.internal_w, r.internal_h, r.confidence);
```

三个入口，按平台已有的数据格式选择，避免额外拷贝：

| 函数 | 输入 | 典型来源 |
| --- | --- | --- |
| `pp_analyze_rgb` | RGB888，3 B/px | 自己解码的截图、RGB Bitmap |
| `pp_analyze_rgba` | RGBA8888，4 B/px，忽略 alpha | Android `ARGB_8888`、HarmonyOS `RGBA_8888` |
| `pp_analyze_luma` | 8-bit 灰度，1 B/px | 视频/NV12 的 Y 平面 |

`stride_bytes` 是**行字节数**，必须不小于 `width × 通道数`。
ROI 传 `pp_roi{x0,y0,x1,y1}`（半开区间），结果仍按整图分辨率换算；不需要 ROI 时传 `NULL`。
`quick` 是保留参数：C ABI 暂不导出逐块热图，因此当前不影响结果，传 0/1 都可以。

### 结果字段

- `method`：`PP_METHOD_NONE` / `PP_METHOD_NEAREST` / `PP_METHOD_SMOOTH`
- `scale`、`scale_x`、`scale_y`：放大倍率
- `internal_w`、`internal_h`：推断的内部分辨率
- `confidence`：0–1
- `bars`：上/下/左/右黑边
- `candidates[]`：常见档位与精确推断，按分数排序
- `evidence_x/y`：每个轴的周期、支持块数、拟合谷深等证据
- `elapsed_ms`：本次分析耗时

ABI 说明：结构体字段只会在主版本升级时做不兼容变更；升级 `.so` 时请同步更新
`pixelpeeper.h`，不要只替换二进制。

## Android 接入

### 1. Gradle / CMake

把 `native/` 拷贝进工程（例如 `app/src/main/cpp/pixelpeeper`），在模块的
`build.gradle.kts` 中：

```kotlin
android {
    defaultConfig {
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DPP_BUILD_CLI=OFF",
                    "-DPP_BUILD_TESTS=OFF",
                    "-DPP_BUILD_JNI=ON"
                )
                cppFlags += "-std=c++17"
            }
        }
    }
    externalNativeBuild {
        cmake { path = file("src/main/cpp/pixelpeeper/CMakeLists.txt") }
    }
}
```

`PP_BUILD_JNI=ON` 会额外生成 `libpixelpeeper_jni.so`，它依赖 `libpixelpeeper.so`。
把 `platform/android/NativeAnalyzer.kt` 放进 `com.pixelpeeper` 包；加载顺序已经在
它的 `init` 中处理。

### 2. 把 Bitmap 变成 direct ByteBuffer

```kotlin
val bmp = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
    decoder.isMutableRequired = false
}
val buffer = ByteBuffer.allocateDirect(bmp.byteCount).order(ByteOrder.nativeOrder())
bmp.copyPixelsToBuffer(buffer)
buffer.rewind()

val result = NativeAnalyzer.analyze(
    rgb = buffer,
    width = bmp.width,
    height = bmp.height,
    roi = intArrayOf(600, 300, 1200, 800),
    rgba = true,
    stride = bmp.rowBytes,
    quick = true
)
```

> `Bitmap.copyPixelsToBuffer` 的通道顺序取决于 `Bitmap.Config`。`ARGB_8888` 在内存中
> 是 RGBA 字节序，因此用 `rgba = true`；如果你先转成 RGB888，再用 `rgba = false`。

## HarmonyOS 接入

### 1. CMake

在 DevEco Studio 的 native 模块 `CMakeLists.txt` 中：

```cmake
set(PP_BUILD_SHARED OFF)
set(PP_BUILD_CLI OFF)
set(PP_BUILD_TESTS OFF)
set(PP_BUILD_NAPI ON)
add_subdirectory(pixelpeeper/native)
```

`PP_BUILD_NAPI=ON` 会把核心和 Node-API 封装一起编成 `libpixelpeeper.so`。
HarmonyOS 通过库名导入，所以产物名必须是 `pixelpeeper`。

### 2. ArkTS 调用

```ts
import pixelpeeper from 'libpixelpeeper.so';

// PixelMap 的 readPixelsToBuffer 得到 RGBA_8888 的 ArrayBuffer
const buffer = new ArrayBuffer(pixelMap.getPixelBytesNumber());
await pixelMap.readPixelsToBuffer(buffer);

const r = pixelpeeper.analyze(
  buffer,
  pixelMap.getImageInfoSync().size.width,
  pixelMap.getImageInfoSync().size.height,
  width * 4,          // stride
  [x0, y0, x1, y1],   // ROI，可传 null
  true,               // quick
  true                // rgba
);
console.info(`${r.internalWidth}x${r.internalHeight}, ${r.scale}x, ${r.confidence}`);
```

返回对象字段为 camelCase，与 Kotlin 版一致：`internalWidth`、`internalHeight`、
`scale`、`method`、`confidence`、`candidates`、`evidenceX` 等。
如果平台给的是 RGB888，把最后一个参数改成 `false`，stride 改成 `width * 3`。

## 为什么 AI 直接转写容易错

Python 版大量依赖 NumPy 的隐式语义，逐行“翻译”时最容易在这些地方出错：

1. **轴的方向**：`gray.T` 不是简单的转置循环。横轴分析按列取差分，纵轴分析按行取差分；
   `_nearest_tiles` 里又对转置后的 tile 调用 `_profiles(t, 1)`，索引顺序错一次就会越界或
   把横纵倍率弄反。
2. **float32 与 float64**：重采样拟合的采样点、插值权重在 Python 里是 `float32`，
   误差均值却用 `float32` 累加；改成全 double 会让谷深和阈值判断漂移。
3. **NumPy 的 percentile/quantile**：默认是线性插值，不是取最近秩。
4. **`np.linspace(...).astype(int)`**：截断而不是四舍五入。
5. **阈值扫描**：`_dup_search` 同时对“唯一值”和 `nextafter` 生成阈值，并依赖
   `bincount` 的周期投票；少一个分支就会漏掉 1.5× 这类非整数倍率。
6. **组合规则**：两个轴证据不一致时取谁、2× 谐波如何压制、近原生 1.07× 如何兜底，
   这些规则比单个检测器更影响最终结果。

本移植保留了这些细节，并用 Python 测试生成同一批像素做逐项对比，而不是只对最终
“看起来差不多”的结果。

## 回归测试

先编译 `pp_cli`，然后：

```bash
python native/tests/test_parity.py -v
```

或指定可执行文件：

```bash
PP_CLI=/path/to/pp_cli python native/tests/test_parity.py -v
```

测试覆盖：2× 最近邻、1.5× 最近邻、JPEG 后的 2×、1.5× 双线性、1.07× 近原生、
ROI 按整图换算、原生不误报、2× 系统放大再套 1.5× 渲染、黑边裁切，以及仓库自带的
真实游戏截图 `docs/demo.png`。

## 已知边界

- DLSS / TSR / FSR2 等时域重建会打散规则网格，通常返回 `none`；
- 截图被聊天软件非整数缩放后，网格已被破坏，任何实现都无法可靠恢复；
- 核心不做图片解码，也不做 UI；它只回答“这张已解码的像素图里有没有放大网格”；
- 极小图（宽或高 < 200）会跳过重采样拟合，只保留重复像素检测；
- 移动端建议先框选 3D 锯齿区域再分析；`quick` 目前只是兼容参数，不改变结果。
