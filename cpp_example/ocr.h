#ifndef OCR_H
#define OCR_H

#include <opencv2/core.hpp>

#include <string>

void InitOcr();
std::string Classification(cv::Mat& originalImage);
void Close();

#endif // OCR_H
