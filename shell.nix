{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.numpy
      python-pkgs.pytest_7
      python-pkgs.pandas
      python-pkgs.pandas-stubs
    ]))
  ];

  shellHook = "
    export PYTHONPATH=$PWD:$PYTHONPATH
  ";
}
