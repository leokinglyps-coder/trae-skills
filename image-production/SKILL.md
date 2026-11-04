---
name: image-production
description: 在本机 ComfyUI 上用三套已验证管线出图（z_image turbo 文生图、z_image 遮罩重绘、Qwen-Image 2.1 文生图/编辑/2K 超分）。当用户要做 图片制作、AI 出图、文生图、局部重绘或往照片里加人、用 Qwen-Image 或 z_image 出图、2K 高清放大时使用。不用于视频生成，也不用于模型下载或删除。
---

# 图片制作（ComfyUI 出图规范）

本机 Windows + RTX 4070 SUPER 12GB 上的出图流程。三套管线均已实测，参数按下表照抄，不要凭经验改。

## 0. 环境

| 项 | 值 |
|---|---|
| ComfyUI | 0.43.0，装在 `D:\AI\ComfyUI-ws` |
| Python / torch | 3.12.15 / 2.9.0+cu129 |
| 显存 / 内存 | 12GB（vram_total 12.88 GB）/ 约 23GiB |
| 启动 | `D:\AI\ComfyUI-ws\scripts\start-comfyui.bat` |
| 关闭 | `D:\AI\ComfyUI-ws\scripts\stop-comfyui.bat` |
| 地址与参数 | `http://127.0.0.1:8188`，`--listen 127.0.0.1 --port 8188 --cache-none` |
| 模型挂载 | `D:\AI\ComfyUI-ws\extra_model_paths.yaml` → `D:\AI\ComfyUI-GGUF\ComfyUI\models`（只挂路径，不拷贝） |
| 出图目录 | `D:\AI\ComfyUI-ws\output\`（`SaveImage` 序号自增，不覆盖旧图） |
| 工作流目录 | `D:\AI\ComfyUI-ws\user\default\workflows\` 才会出现在界面侧边栏 Workflows 面板 |

本技能只操作图片线。不要在该 ComfyUI 上跑视频模型，也不要动视频线的权重与产出。

## 1. 选方案

| 需求 | 方案 | 工作流 | 耗时 | 显存峰值 |
|---|---|---|---|---|
| 快速出概念图，只要 20 秒出图 | z_image turbo 文生图 | `assets/workflows/z_image_t2i.json` | ~20s | ~11.4GB |
| 往现有照片里加人/改一小块，要快、能容忍原主体被改 | z_image 遮罩重绘 | `assets/workflows/z_image_inpaint_mask.json` | ~21s | 同 z_image 量级（未单独记录） |
| 人物一致性优先、要把参考人/物放进画面、要长中文指令 | Qwen-Image 2.1 图像编辑 | `assets/workflows/qwen_edit.json` | ~97s | 11.5–11.8GB |
| 高质量文生图、长中文提示词 | Qwen-Image 2.1 文生图 | `assets/workflows/qwen_t2i.json` | ~49.5s | 未单独记录（同 Qwen 量级） |
| 2K 直出并放大到 3K | Qwen-Image 2.1 2K + 超分 | `assets/workflows/qwen_t2i_2k_upscale.json` | ~87s（2K）/ ~164s（整链） | ~11.47GB |

判据：

- 目标是"改一张已有照片，且要和原片看起来是同一张" → Qwen-Image 2.1 编辑。它整图重算 + 参考图条件注入，光照、比例、景深统一，没有拼贴接缝；代价是比 z_image 局部重绘慢 4–5 倍，显存几乎打满。
- 只要快、只是试构图 → z_image 文生图。
- 从零生成画面、不涉及参考主体 → Qwen 文生图（一致性最好），中文长提示词表现稳定。
- z_image 遮罩重绘只在"接受原主体长相被改动、追求速度"时用。

## 2. 参数表

### 2.1 z_image turbo 文生图

| 项 | 值 |
|---|---|
| 扩散模型 | `z_image_turbo_bf16.safetensors`（11.46GB） |
| CLIP | `qwen_3_4b.safetensors`（7.49GB），`CLIPLoader` type **`lumina2`** |
| VAE | `ae.safetensors`（0.31GB） |
| 尺寸 | 1024×1024（16 的倍数） |
| steps / cfg | 8 / 1.0 |
| sampler / scheduler | `res_multistep` / `simple` |
| 采样偏移 | `ModelSamplingAuraFlow` shift **3** |
| denoise | 1.0（文生图） |
| 负向条件 | `ConditioningZeroOut`（不是手写负向提示词） |
| 实测 | 约 20 秒，显存峰值约 11.4GB |

### 2.2 z_image 遮罩局部重绘（加人进照片）

| 项 | 值 |
|---|---|
| 输入 | 两个 `LoadImage`：原图 + 黑白遮罩 PNG（白=重绘，黑=保留） |
| 画布 | `ImageScale` lanczos → 768×1376（竖向；按目标构图定，必须是 16 的倍数） |
| 遮罩转潜空间 | `ImageToMask` channel **`red`** → `SetLatentNoiseMask` |
| 模型 | 同 2.1（`z_image_turbo_bf16` + `qwen_3_4b`/`lumina2` + `ae`） |
| steps / cfg | 10 / 1.0 |
| sampler / scheduler | `res_multistep` / `simple` |
| denoise | **1.0**（1.0 才整块重绘；降到 0.4–0.6 会改动原主体长相） |
| 实测 | 768×1376、10 步约 21 秒 |

遮罩要点：

- 遮罩全黑 = 没有任何重绘区域，出图和原图一模一样，这不是故障。
- 遮罩最好按缩放后的画布尺寸（本例 768×1376）导出：原图会被 `ImageScale` 缩放，遮罩不走缩放节点，尺寸不一致时边缘会变软。
- 提示词写"要出现什么"（新人物的衣着、站位、光照方向），不要写"修复遮罩"。

### 2.3 Qwen-Image 2.1（主选）

| 项 | 值 |
|---|---|
| 扩散模型 | `qwen_image_2.1_turbo_int8_convrot.safetensors`（6.76GB） |
| 文本编码器 | `qwen3vl_8b_int8_convrot.safetensors`（8.71GB），`CLIPLoader` type **`qwen_image`** |
| VAE | `qwen_image_2.1_vae_bf16.safetensors`（0.63GB） |
| 专用节点 | `QwenImage21Cache`（串在 UNET 之后）、`TextEncodeQwenImage21` |
| steps / cfg | 8 / 1.0（`CFGGuider`） |
| sampler | `KSamplerSelect` = `euler` |
| sigma | `ManualSigmas`，官方 Turbo 8 步表：`1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0` |
| 采样 | `RandomNoise` + `SamplerCustomAdvanced` |
| 文生图尺寸 | `EmptyLatentImage` 1024×1024；`TextEncodeQwenImage21` 的 resolution 与画布一致 |
| 编辑输入 | `LoadImage` → `TextEncodeQwenImage21` 的 `images.image_1` + `vae`；resolution **1280**（参考图归一到 1280） |
| 编辑输出 | 960×1696（9:16），由 resolution 与参考图比例决定 |
| 落盘 | `SaveImageAdvanced`（png / 8-bit / sRGB）或 `SaveImage` |

实测耗时：

| 任务 | 尺寸 | 耗时 | 显存峰值 |
|---|---|---|---|
| 文生图 | 1024×1024 | ~49.5s | 未单独记录 |
| 图像编辑 | 960×1696（参考图归一到 1280） | ~97s | 11.5–11.8GB |
| 2K 直出 | 2048×2048 | ~87s | 11.47GB |
| 3K 整链（2K → 超分 → 回落） | 3072×3072 | ~164s | 同上 |

超过 2K 必须靠超分：`UpscaleModelLoader` 加载 `RealESRGAN_x4plus.pth` → `ImageUpscaleWithModel` → `ImageScale`（lanczos，落到 3072）→ `SaveImage`。这是纯像素放大，不会长出原始画面里没有的真实纹理，别承诺"更真实的细节"。

## 3. 操作步骤

1. 启动服务（`start-comfyui.bat`），bin 窗口保持打开；用 `GET /system_stats` 确认 `vram_free` 足够（跑前关浏览器和游戏）。
2. 载入工作流：从 `assets/workflows/` 复制 JSON 到 `D:\AI\ComfyUI-ws\user\default\workflows\`，刷新侧边栏 Workflows 面板选中；或直接把 JSON 拖进界面。
3. 改三个东西：提示词（`CLIPTextEncode` / `TextEncodeQwenImage21`）、画布尺寸、种子（想复现就固定种子）。
4. 编辑任务额外准备一张清晰参考图，放进 `D:\AI\ComfyUI-ws\input\`，在 `LoadImage` 里选它。
5. 点 Run，等任务队列跑完。
6. 按第 6 节验收，不要只看"任务成功"。

## 4. 提示词

结构：主体 + 姿态 + 环境 + 光线 + 镜头 + 质感。中文可直接用（Qwen 系列中文理解好），z_image 中英文都行。

已验证示例（海边美女，Qwen-Image 2.1 2K 出图）：

```
清晨的海边，薄雾弥漫在空气中，一位年轻亚洲女性站在齐踝的浅水里，身穿白色轻薄的长裙，赤脚，
长发和裙摆被海风轻轻吹起，姿态自然放松，侧身微微回望镜头，神情安静柔和；远处海面、礁石和海岸线
在晨霭里一层层淡出，两三只海鸥在她身后低空掠过，最近的一只贴着水面飞行，没有遮住人物；柔和的逆光
穿过雾气形成淡淡的光晕，低对比，空气中有细腻的雾气层次；人物面部、发丝和裙料对焦精准、清晰锐利，
皮肤有真实质感，织物纹理清楚，只有远景因雾气而变淡；真实摄影，85mm 中长焦，f/2 浅景深，柔和高光，
电影感，高细节
```

编辑加人示例（已验证）：

```
在照片中男士的右侧添加一位年轻亚洲女性一起合照，她面向镜头微笑，穿着与场合相符的浅色连衣裙，
身高和透视与男士一致，光照方向、色温、白平衡与原照片完全一致，背景和地面保持原样
```

完整写法、负向提示词、按模型差异调整见 `references/prompt-guide.md`。

## 5. 验收

- 到 `D:\AI\ComfyUI-ws\output\` 找新文件，读出**尺寸**和**字节数**，确认不是 0 字节、不是异常小尺寸。
- 看显存峰值是否在 11.4–11.8GB 区间；超了就说明参数被改大或后台有其他程序。
- 在浏览器实机载入同一份 workflow，确认没有红框、没有"缺节点"报错。
- 记录提示词、尺寸、种子、denoise（复现和调优都要靠这些）。

## 6. 故障对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `CLIPLoader` 节点红框 / 加载报错 | type 填错 | z_image 必须 `lumina2`，Qwen 必须 `qwen_image` |
| OOM / 显存不足 | 尺寸超 2K，或后台浏览器/游戏占显存 | 降到 2048×2048 以内；关浏览器；不要加 `--lowvram`（实测不需要，加了只会变慢） |
| 遮罩重绘出图和原图一样 | 遮罩全黑，或遮罩与画布尺寸不匹配 | 检查白色区域；按目标画布尺寸重导遮罩 |
| 局部重绘后原主体长相变了 | denoise 低于 1.0 | 提到 1.0；要保原样改用 Qwen 编辑 |
| 出图糊、细节发虚 | 边长低于 640，或提示词没有质感/镜头描述 | 提到 1024 级别；补对焦、材质、镜头描述 |
| 尺寸报错或结果异常 | 宽高不是 16 的倍数 | 取最近的 16 倍数 |
| 竖向大图必 OOM | 超过 1024×1800 竖向 | 压低到 768×1376 或 1024×1536 |
| 模型下载只有 0.2–0.5MB/s | 走了 hf-mirror | 换 `modelscope.cn/models/Comfy-Org/Qwen-Image-2.1`，实测 10–20MB/s |
| 换了模型后整个流程崩 | UNET / CLIPLoader(type) / VAE 没成套 | 三者必须成套配对，不要单独换其中一个 |
| 工作流不在侧边栏 | 没放对目录 | 放 `D:\AI\ComfyUI-ws\user\default\workflows\` |
| 旧图被覆盖 | 以为 SaveImage 会覆盖 | 实际序号自增不覆盖；找不到图就按前缀查 output 目录 |

## 7. 硬限制

- 12GB 显存是硬边界：2K（2048×2048）是直出上限，超过必 OOM；更高只能超分。
- 分辨率必须是 16 的倍数；低于 640 会糊；竖向超过 1024×1800 易爆显存。
- 不要盲目加 `--lowvram`。
- `UNETLoader` / `CLIPLoader`(type) / `VAELoader` 必须成套配对。
- 编辑任务优先 Qwen 编辑，不要用 z_image 局部重绘去要求"保持原人物一模一样"。

## 8. 文件

| 路径 | 用途 |
|---|---|
| `assets/workflows/z_image_t2i.json` | z_image turbo 文生图，1024×1024、8 步 |
| `assets/workflows/z_image_inpaint_mask.json` | z_image 遮罩局部重绘，768×1376、10 步、denoise 1.0 |
| `assets/workflows/qwen_t2i.json` | Qwen-Image 2.1 文生图，1024×1024 |
| `assets/workflows/qwen_edit.json` | Qwen-Image 2.1 图像编辑（加人/改图） |
| `assets/workflows/qwen_t2i_2k_upscale.json` | 2K 直出 + Real-ESRGAN 超分到 3072 |
| `references/prompt-guide.md` | 出图提示词写法与已验证示例 |
