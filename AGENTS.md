# 仓库指南

## 项目结构与模块组织

- `eas/`：核心库
  - `eas/mk.py`：Python DSL 原语（仅追踪；无急切执行）
  - `eas/kernel.py`：`@eas.kernel` 装饰器，追踪器 → IR 构建器，内核缓存
  - `eas/ir.py`：最小化、Metal无关的 IR
  - `eas/codegen/msl.py`：IR → Metal Shading Language (MSL) 代码生成（1D MVP）
  - `eas/runtime/`：后端（`cpu.py`、`metal.py`）和 ObjC++ 扩展源码（`_metal.mm`）
- `examples/`：可运行的演示（如 `python3 -m examples.add`）
- `tests/`：单元测试（`unittest`）
- `tools/`：开发者脚本（如 `tools/build_metal_ext.py`）

## 构建、测试和开发命令

- `python3 -m unittest discover -s tests -p "test*.py" -v`：运行测试套件
- `python3 -m examples.add`：运行元素级加法演示（使用 `EAS_BACKEND=auto`）
- `python3 tools/build_metal_ext.py`：构建 ObjC++ 扩展 `eas._metal`（需要 macOS + Xcode SDK）
- `EAS_BACKEND=metal python3 -m examples.add`：强制使用 Metal 运行时
- `EAS_BACKEND=cpu python3 -m examples.add`：强制使用 CPU 运行时

## 编码风格与命名约定

- Python：4 空格缩进，优先使用类型提示，保持代码简洁可读（PEP 8 风格）
- 命名：内核使用 `*_kernel`；编译时元参数使用全大写（如 `BLOCK: eas.constexpr`）
- 添加新的 DSL 操作时，需在完整流水线中一致更新：`eas/mk.py` → IR 构建器（`eas/kernel.py`） → 代码生成（`eas/codegen/msl.py`） → 运行时（`eas/runtime/*`）

## 测试指南

- 框架：`unittest` + `numpy.testing` 用于数值断言
- 位置/模式：`tests/test_*.py`
- 如果添加Metal特定测试，保持 CPU 基线并在使用 `eas._metal`/设备不可用时跳过/保护 Metal 运行

# 额外指令

- 项目使用uv作为包管理器
- codex运行在sandbox内，如果需要运行性能测试，通知用户来进行