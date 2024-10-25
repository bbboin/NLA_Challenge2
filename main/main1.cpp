#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Eigenvalues> 
#include <Eigen/SVD>

#include "../include/myl4image.h" //also include stb_image.h and stb_image_write.h

using namespace std;
using namespace Eigen;

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main(int argc, char* argv[]) {

  ////////////////////////////////////////////////////////////////1
  //Error if forget image path
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <image_path>" << endl;
    return 1;
  }
  //assign image path
  const char* input_image_path = argv[1];
  // Load the image using stb_image
  int width, height, channels;
  // Force load as B&W
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
  //error loading image
  if (!image_data) {
    cerr << "Error: Could not load image " << input_image_path << endl;
    return 1;
  }
  //print image parameters
  cout << endl << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl << endl;
  // Prepare Eigen matrix for B&W channel
  MatrixXd A(height, width);
  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * 1;  // 1 channels (B&W)
      A(i, j) = static_cast<double>(image_data[index])/255.0;
    }
  }
  // Free memory
  stbi_image_free(image_data);
  //compute A'A
  auto ATA = A.transpose()*A;
  //norm
  cout << "norm of A'A:   " << ATA.norm() << endl << endl;
  
  ////////////////////////////////////////////////////////////////2
  //eigenvalue problem
  SelfAdjointEigenSolver<MatrixXd> es(ATA);
  if (es.info() != Eigen::Success) abort();
  auto eig = es.eigenvalues();
  //singular value
  VectorXd sing = eig.array().sqrt();
  //Two largest SV
  cout << "Two largest singular values:   " << endl << sing.reverse().head(2) << endl << endl;
  
  ////////////////////////////////////////////////////////////////3
  //export .mtx
  saveMarket(ATA, "./ATA.mtx");
  //compare with lis
  system("mpirun -n 4 ./etest1 ATA.mtx evector.txt hist.txt -e pi -etol 1.0e-8");

  ////////////////////////////////////////////////////////////////4
  //accelerate with shift
  system("mpirun -n 4 ./etest1 ATA.mtx evector_shift.txt hist_shift.txt -e pi -etol 1.0e-8 -shift 697.0");

  ////////////////////////////////////////////////////////////////5
  //compute SVD
  BDCSVD<MatrixXd> svd;
  svd.compute(A, ComputeThinU | ComputeThinV);
  auto S = svd.singularValues();
  //norm
  cout << "norm of Sigma:   " << S.norm() << endl << endl;

  ///////////////////////////////////////////////////////////////6
  constexpr int k1 = 40;
  constexpr int k2 = 80;
  //compute C, D
  MatrixXd C1 = svd.matrixU().leftCols(k1);
  MatrixXd C2 = svd.matrixU().leftCols(k2);
  MatrixXd D1 = svd.matrixV().leftCols(k1) * static_cast<MatrixXd>(S.asDiagonal()).topLeftCorner(k1, k1);
  MatrixXd D2 = svd.matrixV().leftCols(k2) * static_cast<MatrixXd>(S.asDiagonal()).topLeftCorner(k2, k2);
  //nonzeros
  cout << "Number of non-zeros entries in C, k=40:   " << C1.nonZeros() << endl;
  cout << "Number of non-zeros entries in D, k=40:   " << D1.nonZeros() << endl;
  cout << "Number of non-zeros entries in C, k=80:   " << C2.nonZeros() << endl;
  cout << "Number of non-zeros entries in D, k=80:   " << D2.nonZeros() << endl << endl;

  ///////////////////////////////////////////////////////////////7
  //compute CD'
  MatrixXd A1 = C1*D1.transpose();
  MatrixXd A2 = C2*D2.transpose();
  //export
  limit01(A1, height, width);
  save_image(A1, height, width, "A40.png");
  limit01(A2, height, width);
  save_image(A2, height, width, "A80.png");

  ///////////////////////////////////////////////////////////////8
  //create checkerboard
  constexpr int n = 200;
  MatrixXd board = MatrixXd::Zero(n,n);
  for (int i=0; i<n; i++){
    for (int j=0; j<n; j++){
      if ((static_cast<int>(i/(n/8))+static_cast<int>(j/(n/8)))%2) board(i,j) = 1. ;
    }
  }
  //norm
  cout << endl << "norm of checkerboard:   " << board.norm() << endl << endl;

  ///////////////////////////////////////////////////////////////9
  //generate and adding noise
  MatrixXd noisy = board + MatrixXd::Random(n, n)*50.0/255.0;
  //export
  limit01(noisy, n, n);
  save_image(noisy, n, n, "noisy.png");

  ///////////////////////////////////////////////////////////////10
  //compute SVD
  BDCSVD<MatrixXd> svd_noisy;
  svd_noisy.compute(noisy, ComputeThinU | ComputeThinV);
  auto S_noisy = svd_noisy.singularValues();
  //Two largest SV
  cout << endl << "Two largest singular values of noisy:   " << endl << S_noisy.head(2) << endl << endl;

  ///////////////////////////////////////////////////////////////11
  constexpr int k3 = 5;
  constexpr int k4 = 10;
  //compute C, D
  MatrixXd C1cb = svd_noisy.matrixU().leftCols(k3);
  MatrixXd C2cb = svd_noisy.matrixU().leftCols(k4);
  MatrixXd D1cb = svd_noisy.matrixV().leftCols(k3) * static_cast<MatrixXd>(S_noisy.asDiagonal()).topLeftCorner(k3, k3);
  MatrixXd D2cb = svd_noisy.matrixV().leftCols(k4) * static_cast<MatrixXd>(S_noisy.asDiagonal()).topLeftCorner(k4, k4);
  cout << "Size of C, k=5:   " << C1cb.rows() << "x" << C1cb.cols() << endl;
  cout << "Size of D, k=5:   " << D1cb.rows() << "x" << D1cb.cols() << endl;
  cout << "Size of C, k=10:   " << C2cb.rows() << "x" << C2cb.cols() << endl;
  cout << "Size of D, k=10:   " << D2cb.rows() << "x" << D2cb.cols() << endl << endl;
  
  ///////////////////////////////////////////////////////////////12
  //compute CD'
  MatrixXd A1cb = C1cb*D1cb.transpose();
  MatrixXd A2cb = C2cb*D2cb.transpose();
  //export
  limit01(A1cb, n, n);
  save_image(A1cb, n, n, "cb5.png");
  limit01(A2cb, n, n);
  save_image(A2cb, n, n, "cb10.png");

  return 0;
}