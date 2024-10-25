#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//typedef Eigen::Triplet<double> T;


//function that save image as .png
int save_image(Eigen::MatrixXd const & A, int const height, int const width, std::string const output_image_path) {

  //Prepare matrix for export
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> image(height, width);
  // Use Eigen's unaryExpr to map the values (0.0 to 1.0) to 0 to 255
  image = A.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val*255.0);
  });

  // Save the image using stb_image_write
  if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                     image.data(), width) == 0) {
    std::cerr << "Error: Could not save image: " << output_image_path << std::endl;
    
    return 1;
  }

  std::cout << "Image saved to " << output_image_path << std::endl;

  return 0;

}

//function that limit values between 0,1
Eigen::MatrixXd limit01 (Eigen::MatrixXd & A, int const height, int const width) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (A(i,j)<0.0) {
        A(i,j)=0.0;
      }
      if (A(i,j)>1.0) {
        A(i,j)=1.0;
      }
    }
  }
  return A;
}

//function that build convolution matrix from 3x3 kernel
Eigen::SparseMatrix<double,Eigen::RowMajor> ker2mat (std::array<std::array<double,3>,3> const & kernel, int const height, int const width) {
  Eigen::SparseMatrix<double,Eigen::RowMajor> A(height*width,height*width);
  
  std::vector<Eigen::Triplet<double>> tripletList;

  tripletList.reserve((height*3-2)*(width*3-2));
  //---------------------------------------------------------------can do better
  for(int i=0; i<height*width; i++) {
    if (kernel[1][1]) tripletList.push_back(Eigen::Triplet<double>(i, i, kernel[1][1]));
    if ((i+1)%height) {
      if (kernel[0][1])tripletList.push_back(Eigen::Triplet<double>(i+1, i, kernel[0][1]));
      if (kernel[2][1])tripletList.push_back(Eigen::Triplet<double>(i, i+1, kernel[2][1]));
    }
    if (i<height*(width-1)) {
      if (kernel[1][0]) tripletList.push_back(Eigen::Triplet<double>(i+height, i, kernel[1][0]));
      if (kernel[1][2]) tripletList.push_back(Eigen::Triplet<double>(i, i+height, kernel[1][2]));
      if ((i+1)%height) {
        if (kernel[0][0]) tripletList.push_back(Eigen::Triplet<double>(i+height+1, i, kernel[0][0]));
        if (kernel[2][2]) tripletList.push_back(Eigen::Triplet<double>(i, i+height+1, kernel[2][2]));
      }
      if (i%height) {
        if (kernel[2][0]) tripletList.push_back(Eigen::Triplet<double>(i+height-1, i, kernel[2][0]));
        if (kernel[0][2]) tripletList.push_back(Eigen::Triplet<double>(i, i+height-1, kernel[0][2]));
      }
    }
  }

  A.setFromTriplets(tripletList.begin(), tripletList.end());
  return A;
}

