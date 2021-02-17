// Arjun Ramaswami
#include <iostream>
#include <string>
#include "cxxopts.hpp" // Cmd-Line Args parser
#include "conv3D.hpp"

void
parse_args(int argc, char* argv[], std::string& inp_fname, std::string& out_fname){

  try{
    cxxopts::Options options("Convolution3D", "3D Conv Filter on incoming images");
    options.add_options()
        ("f, file", "Input Filename", cxxopts::value<std::string>())
        ("o, out", "Output Filename", cxxopts::value<std::string>()->default_value("a.out"))
        ("h,help", "Print usage")
    ;
    auto opt = options.parse(argc, argv);

    // print help
    if (opt.count("help")){
      std::cout << options.help() << std::endl;
      exit(0);
    }

    if(opt.count("file")){
      inp_fname = opt["file"].as<std::string>();
    }
    else{
      std::cout << "\tPlease Input a file" << std::endl;
      exit(1);
    }
    if(opt.count("out")){
      out_fname = opt["out"].as<std::string>();
      std::cout << "Using default output filename - a.out" << std::endl;
    }
  }
  catch(const cxxopts::OptionException& e){
    std::cerr << "Error parsing options: " << e.what() << std::endl;
    exit(1);
  }

}

int main(int argc, char* argv[]){

  std::string inp_fname, out_fname;

  parse_args(argc, argv, inp_fname, out_fname);

  //conv3D();

  // Verify convolution with library

  return 0;
}