{
  description = "Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        
        # Настраиваем Rust (stable с компонентами)
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"    # Для RLS
            "rust-analyzer"
            "clippy"
            "rustfmt"
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustToolchain  # Rust с компонентами
            cargo-edit    # `cargo add` и др.
            cargo-watch   # Автоматический перезапуск
            bacon         # Альтернатива cargo-watch
            rust-analyzer # LSP-сервер
            lldb          # Отладчик
            pkg-config    # Для нативных зависимостей
            openssl       # Часто требуется для крейтов
            glfw
            xorg.libX11
            wayland
            vulkan-loader
            vulkan-tools
            vulkan-headers
            vulkan-validation-layers
            shader-slang
            tokei # cargo utilites
          ];

          # Переменные окружения для Rust
          RUST_BACKTRACE = "1";
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib:${pkgs.glfw}/lib";
          shellHook = ''
            echo "Rust stable: $(rustc --version)"
            echo "Cargo: $(cargo --version)"
            echo "Vulkan loader: $(find ${pkgs.vulkan-loader}/lib -name libvulkan.so*)"
            vulkaninfo | grep "Vulkan API version"
          '';
        };
      }
    );
}
