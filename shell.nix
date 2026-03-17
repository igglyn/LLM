# I am a devshell enjoyer, however CUDA doesn't work

# Run with `nix-shell cuda-shell.nix`
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
   name = "shell";
   buildInputs = with pkgs; [
     python313
     python313Packages.pytest
     python313Packages.torch
     python313Packages.transformers
     python313Packages.datasets
   ];
}
