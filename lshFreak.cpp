
#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <cstdlib>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/flann/flann.hpp"
#include "lsh_index.h"

#include <ctime>
using namespace cv;

void readme();
bool hasEnoughDistance(Point a, Point b);
float computeDistance(Point a, Point b);
#include <math.h>

typedef struct _ImgCount {
	unsigned long imgid;
	int count;
} ImgCount;

int compareImgCount(const void *a, const void *b){
	ImgCount *x = (ImgCount*)a;
	ImgCount *y = (ImgCount*)b;
	return(y->count > x->count) - (y->count < x->count);
}

int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1;}

	std::cout<<"size of int: "<<sizeof(int)<<std::endl;
	DIR *pDIR;
	struct dirent *entry;
	std::vector<Mat> descriptors;
	std::vector<std::string> imgNames;
	if( pDIR=opendir(argv[1]) ){
		int num_img = 0;
		unsigned int num_des = 0;
		std::cout<<"Reading data points..."<<std::endl;
		clock_t start = clock();
		while(entry = readdir(pDIR)){
			
			if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){
				string s1 = argv[1];
				string s2 = entry->d_name;
				std::string img_path = s1 + '/' + s2;
				std::cout<<"Reading img "<<num_img<<": "<<s2<<std::endl;
				
				Mat img_1 = imread( img_path.data(), CV_LOAD_IMAGE_GRAYSCALE );
			
				if( !img_1.data  )
				  { 
					std::cout<< " --(!) Error reading images " << std::endl; 
					continue;
					//return -1; 
				  }
				  imgNames.push_back(s2);
				  //-- Step 1: Detect the keypoints
				  int minHessian = 400;
				  //SurfFeatureDetector detector( minHessian );
				  OrbFeatureDetector detector( minHessian );
				  std::vector<KeyPoint> keypoints_1;
				  detector.detect( img_1, keypoints_1 );
				  //std::cout <<  num_img<<" keypoints.size: " <<  keypoints_1.size()<< std::endl;
				  //std::cout <<  " coordinate: " <<  keypoints_1[1].pt << std::endl;




				  //-- Step 2: Calculate descriptors (feature vectors)
				  //SurfDescriptorExtractor extractor;
				  //OrbDescriptorExtractor extractor;
				  FREAK extractor;
				  Mat descriptors_1;
				  extractor.compute( img_1, keypoints_1, descriptors_1 );
				  
				  descriptors.push_back(descriptors_1);
				  //std::cout <<num_img<<  " descriptors rows: " <<  descriptors_1.rows<< std::endl;
				  //std::cout <<num_img<<" descriptors cols: " <<  descriptors_1.cols<< std::endl;
				  //for( int i = 0; i < descriptors_1.rows; i++ )
				  //{
					//  for( int j = 0; j < descriptors_1.cols; j++ ){
					//	unsigned char var = descriptors_1.at<unsigned char>(i,j);
					//	int des = var;
					//	 std::cout <<(double)des/1000<<" ";
					 // }
					  //std::cout<<num_img<<std::endl;
				  //}
				  num_des += descriptors_1.rows;
				  num_img++;
			}
		}
		clock_t end = clock();
	    double diffms    = (end-start) / ( CLOCKS_PER_SEC / 1000 );
	    std::cout<<"Complete! "<<diffms<<"ms"<<std::endl;
		unsigned long total_size = 0;
		for (int i = 0; i < descriptors.size(); i++)
			total_size +=descriptors[i].total();
		std::cout<<"descriptor size: "<<total_size<<std::endl;      
		closedir(pDIR);
		
		//create index with flann lsh
		std::cout<<"Building LSH index...";
	    start = clock();
		unsigned int* desMap = new unsigned int[num_des];
		uchar* matrixData = new uchar[num_des*descriptors[0].cols];
		
		unsigned int temID = 0;
		unsigned int desid = 0;
		for (int i = 0; i < descriptors.size(); i++){
			unsigned int* ids = new unsigned int[descriptors[i].rows];
			for (int j = 0; j < descriptors[i].rows; j++){
				for (int k = 0; k < descriptors[i].cols; k++){
					matrixData[temID] = descriptors[i].at<unsigned char>(j,k);
					temID++;
				}
				desMap[desid] = i;
				desid++;
			}
		}
		descriptors.clear();
		cvflann::Matrix<unsigned char> dataset(matrixData, num_des, descriptors[0].cols);
		cvflann::LshIndex<cvflann::Hamming<unsigned char> > lsh_index(dataset, cvflann::LshIndexParams(15, 24, 1));
		lsh_index.buildIndex();
		//lsh_index.initIndex();
		end = clock();
	    diffms    = (end-start) / ( CLOCKS_PER_SEC / 1000 );
	    std::cout<<"Complete!"<<diffms<<"ms, used memory: "<<lsh_index.usedMemory()<<std::endl;
		std::cout<<"Please type GO to continue:"<<std::endl;
		string str;
		std::cin>>str;
		
		//for (int i = 0; i < descriptors.size(); i++){
		//	unsigned int* ids = new unsigned int[descriptors[i].rows];
		//	for (int j = 0; j < descriptors[i].rows; j++){
		//		desMap[temID] = i;
		//		ids[j] = temID;
		//		temID++;				
		//	}
		//	cvflann::Matrix<unsigned char> dataset(descriptors[i].data, descriptors[i].rows, descriptors[i].cols);
		//	lsh_index.buildIndex(ids, dataset);
		//	delete ids;
		//}
		
		//query
		if( pDIR=opendir(argv[2]) ){
			int num_img_q = 0;
			double hits = 0;
			//hard code 20
			double percentage[20];
			while(entry = readdir(pDIR)){
				if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){
					string s1 = argv[2];
					string s2 = entry->d_name;
					std::string img_path = s1 + '/' + s2;
					Mat img_2 = imread( img_path.data(), CV_LOAD_IMAGE_GRAYSCALE );
					if( img_2.data  ){
						int minHessian = 400;


						  //SurfFeatureDetector detector( minHessian );
						  OrbFeatureDetector detector( minHessian );

						  std::vector<KeyPoint> keypoints_2;

						  detector.detect( img_2, keypoints_2 );

						  //BriefDescriptorExtractor extractor;
						  //OrbDescriptorExtractor extractor;
						  FREAK extractor;

						  Mat descriptors_2;
						  extractor.compute( img_2, keypoints_2, descriptors_2 );
						  //std::cout<<"Number of descriptors: "<<descriptors_2.rows<<std::endl;
						  //lsh search
						  int n = 1;
						  int* resultInd = new int[descriptors_2.rows*n];
						  int* resultDis = new int[descriptors_2.rows*n];
						  cvflann::Matrix<unsigned char> dataset(descriptors_2.data, descriptors_2.rows, descriptors_2.cols);
						  cvflann::Matrix<int> retind(resultInd, descriptors_2.rows, n);
						  cvflann::Matrix<int> retdes(resultDis, descriptors_2.rows, n);
						  std::cout<<std::endl<<"querying "<<s2;
						  start = clock();
						  lsh_index.knnSearch(dataset, retind, retdes, n, cvflann::SearchParams());
						  
						 // for (int i = 0 ; i < retind.rows; i++){
						 //	std::cout<<"des "<<retind[i][0]<<": "<<retdes[i][0]<<std::endl;
						 //}
						  vector<ImgCount> imgcounts;
						  double avgdist = 0;
						  int discounts = 0;
						  for (int i = 0; i < descriptors_2.rows*n; i++){
						    
							bool add = false;
							if (resultDis[i] <= 65){
								avgdist += resultDis[i];
								discounts++;
								for (int imgs = 0; imgs < imgcounts.size(); imgs++){
									// std::cout<<resultInd[i]<<std::endl;
								  if (imgcounts[imgs].imgid == desMap[resultInd[i]]){
									  imgcounts[imgs].count++;
									  add = true;
								  }
								}
								if (!add){
								  ImgCount ic;
								  ic.count = 1;
								  ic.imgid = desMap[resultInd[i]];
								  imgcounts.push_back(ic);
								}
							}
						  }
						  std::cout<<" avg dist: "<<avgdist/discounts;
						  ImgCount* counts = imgcounts.data();
						  qsort(counts, imgcounts.size(), sizeof(ImgCount), compareImgCount);
						  end = clock();
						  diffms    = (end-start) / ( CLOCKS_PER_SEC / 1000 );
						  std::cout<<" Complete!"<<diffms<<"ms"<<std::endl;
						  
						  double num_hit = 0;
						  double num_hits = 0;
						  int top = imgcounts.size() > 3 ? 3 : imgcounts.size();
						  for (int i = 0; i < top; i++){
							
							std::cout<<"Img id"<<counts[i].imgid<<" name: "<<imgNames[counts[i].imgid]<<": "<<counts[i].count<<std::endl;
							if (i == 0)
								num_hit = counts[i].count;
							num_hits += counts[i].count;
						  }
						  if (imgcounts.size() > 0 && imgNames[counts[0].imgid].data()[0] == s2.data()[0] && imgNames[counts[0].imgid].data()[1] == s2.data()[1] && imgNames[counts[0].imgid].data()[2] == s2.data()[2] && imgNames[counts[0].imgid].data()[3] == s2.data()[3])
							hits += 1;
						  else
							std::cout<<"Not Hit"<<std::endl;
						  percentage[num_img_q] = num_hit/num_hits;
						  delete resultInd;
						  delete resultDis;
						 
						num_img_q++;
					} // end of img_2.data
				} //end of if not ./ or ../
			}//end of while
			//while(1){}

			std::cout<<"accuracy: "<<hits/num_img_q<<std::endl;
		} //end of if open dir
    }
	else{
		std::cout<<"Fail to open"<<std::endl;
	}
	
  return 0;
}

void readme()
{
	std::cout << " Usage: extract [folder]" << std::endl;
}

float computeDistance(Point a, Point b)
{
	return sqrt(pow((double)abs(a.x - b.x), 2) + pow((double)abs(a.y - b.y), 2)) ;
}

bool hasEnoughDistance(Point a, Point b)
{
    std::cout << "distance between "<< a << " and "<< b << " is " << sqrt(pow((double)abs(a.x - b.x), 2) + pow((double)abs(a.y - b.y), 2))<<std::endl;
	// assume the minimun target in user's picture is 120*120
	if (sqrt(pow((double)abs(a.x - b.x), 2) + pow((double)abs(a.y - b.y), 2)) > 120.0)
	{
		return true;
	}
	else 
	{
		return false;
	}
}

int hammingDistance(unsigned char* a, unsigned char* b, int size){
	int ret = 0;
	for (int i = 0; i < size; i++){
		ret += a[i]^b[i];
	}
	return ret;
}
