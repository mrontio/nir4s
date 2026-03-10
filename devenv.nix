{ pkgs, lib, config, inputs, ... }:
{
  env = {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.boost
      pkgs.glibc.dev
    ];
    CPATH = lib.makeIncludePath [
      pkgs.boost
      pkgs.glibc.dev
    ];
  };

  packages = [
    pkgs.git
    pkgs.git-lfs
    pkgs.stdenv.cc
    pkgs.boost
    pkgs.metals
    pkgs.scalafmt
    pkgs.python3Packages.jupytext
    pkgs.hdf5
  ];

  languages.java = {
    jdk.package = pkgs.jdk17;
  };

  languages.scala = {
    enable = true;
    package = pkgs.scala_2_13;
    sbt.enable = true;
  };

  languages.python = {
    enable = true;
    venv = {
      enable = true;
      requirements = ''
        --extra-index-url https://download.pytorch.org/whl/cpu
        torch==2.10.0+cpu
        norse==1.1.0
        nir==1.0.4
        numpy==1.26.4
        matplotlib==3.10.8
        tonic==1.6.0
        sinabs==3.1.3
        seaborn==0.13.2
      '';
    };
  };

  git-hooks = {
    default_stages = [ "commit" ];
    hooks = {
      scalafmt = {
        enable = true;
        description = "Format Scala code";
        entry = "${pkgs.scalafmt}/bin/scalafmt";
        files = "\\.scala$";
        language = "system";
      };
    };
  };

  enterTest = "sbt 'testOnly nir.NIRSpec'";


}
