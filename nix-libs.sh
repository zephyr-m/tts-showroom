# NixOS: колёса pip не видят libstdc++ и zlib.
_here=$(dirname "${BASH_SOURCE[0]}")
_gcc_lib=$(dirname "$(g++ -print-file-name=libstdc++.so.6)")
_zlib_out=
if [ -f "$_here/.nix-libz-path" ]; then
    _zlib_out=$(cat "$_here/.nix-libz-path")
fi
if [ ! -f "${_zlib_out}/lib/libz.so.1" ]; then
    _zlib_out=$(nix-build '<nixpkgs>' -A zlib --no-out-link)
    printf '%s\n' "$_zlib_out" > "$_here/.nix-libz-path"
fi
export LD_LIBRARY_PATH="${_gcc_lib}:${_zlib_out}/lib:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _here _gcc_lib _zlib_out
