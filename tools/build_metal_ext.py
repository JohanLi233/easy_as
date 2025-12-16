# filename: tools/build_metal_ext.py

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sysconfig


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "eas" / "runtime" / "_metal.mm"
    if not src.exists():
        raise FileNotFoundError(src)

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("sysconfig EXT_SUFFIX is empty")
    out = root / "eas" / f"_metal{ext_suffix}"

    include = sysconfig.get_path("include")
    platinclude = sysconfig.get_path("platinclude")
    if not include:
        raise RuntimeError("sysconfig include path is empty")

    clangxx = shutil.which("clang++")
    try:
        clangxx = subprocess.check_output(
            ["xcrun", "--find", "clang++"], text=True
        ).strip()
    except Exception:
        pass
    if not clangxx:
        raise RuntimeError("clang++ not found (tried xcrun and PATH)")

    sdk = None
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
        ).strip()
    except Exception:
        pass

    cmd = [
        clangxx,
        "-O3",
        "-Wall",
        "-std=c++17",
        "-fobjc-arc",
        "-bundle",
        "-undefined",
        "dynamic_lookup",
    ]
    if sdk:
        cmd += ["-isysroot", sdk]
    cmd += [
        "-I",
        include,
    ]
    if platinclude and platinclude != include:
        cmd += ["-I", platinclude]

    cmd += [
        "-o",
        str(out),
        str(src),
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        "-framework",
        "CoreFoundation",
    ]

    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=root)
    print(f"built: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
