// Test.cpp : �������̨Ӧ�ó������ڵ㡣
//
#include "utils.h"
#include "stdafx.h"
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cv.h>
#include "surflib.h"
#include "kdtree.h"
void drawRect(IplImage*img,CvRect rect)		
{
    cvRectangle(img,cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width,rect.y+rect.height),cvScalar(255,255,255),1);
}
void sum_rgb(IplImage*src,IplImage* dst){

	IplImage* r=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	IplImage* g=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	IplImage* b=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	cvSplit(src,r,g,b,NULL);
	//IplImage* s=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	cvAddWeighted(r,1./3.,g,1./3.,0.0,dst);
	cvAddWeighted(dst,1./3.,b,1./3.,0.0,dst);
	//cvThreshold(s,dst,180,255,CV_THRESH_BINARY);
	cvReleaseImage(&r);
	cvReleaseImage(&g);
	cvReleaseImage(&b);
	//cvReleaseImage(&s);
}

void mixshow0(IplImage* img0,IplImage* img1)
{
	IplImage* mix = cvCreateImage(cvSize((img0->width+img1->width),MAX(img0->height,img1->height)),IPL_DEPTH_8U,3);
	CvRect rect=cvRect(0,0,img0->width,img0->height);
	cvSetImageROI(mix,rect);
	cvCopy(img0,mix);
	cvResetImageROI(mix);
	rect=cvRect(img0->width,0,img1->width,img1->height);
	cvSetImageROI(mix,rect);
	cvCopy(img1,mix);
	cvResetImageROI(mix);
	cvNamedWindow("mix_0",CV_WINDOW_AUTOSIZE);
	cvShowImage("mix_0",mix);
};

void mixshow1(IplImage* img0,IplImage* img1)
{
	IplImage* mix = cvCreateImage(cvSize((img0->width+img1->width),MAX(img0->height,img1->height)),IPL_DEPTH_8U,1);
	CvRect rect=cvRect(0,0,img0->width,img0->height);
	cvSetImageROI(mix,rect);
	cvCopy(img0,mix);
	cvResetImageROI(mix);
	rect=cvRect(img0->width,0,img1->width,img1->height);
	cvSetImageROI(mix,rect);
	cvCopy(img1,mix);
	cvResetImageROI(mix);
	cvNamedWindow("mix_1",CV_WINDOW_AUTOSIZE);
	cvShowImage("mix_1",mix);
};
int _tmain(int argc, _TCHAR* argv[])
{
	CvSeq *contour = 0; 
	CvSeq *contmax = 0;
	CvSeq *contour2 = 0; 
	CvSeq *contmax2 = 0;
	CvRect aRect;
	CvRect aRect2;
	IpVec ipts1, ipts2;
	IpPairVec matches;

	CvMemStorage * storage = cvCreateMemStorage(0); 
	CvMemStorage * storage2 = cvCreateMemStorage(0);  


	
	//sum_rgb(imgl,gray);
	//sum_rgb(imgr,gray2);

	//�Ե�һ��ͼ�Ĵ���
	IplImage* imgl = cvLoadImage("E:\\test2.jpg");//��ȡ
	IplImage *gray = cvCreateImage(cvGetSize(imgl), IPL_DEPTH_8U, 1);//�����Ҷ�ͼ
	cvCvtColor(imgl, gray, CV_RGB2GRAY);//תΪ�Ҷ�ͼ
	cvSmooth(gray,gray, CV_GAUSSIAN ,5,5);//�����˲�ȥ�����
	cvNormalize(gray,gray, 1, 255, CV_MINMAX, NULL);	//ͼ���һ��
	cvAdaptiveThreshold(gray,gray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,3,2);//��ȡ����Ӧ��ֵ�ķ���
		cvErode(gray,gray,0,2);//��ʴ����
	cvDilate(gray,gray,0,2);//�������Σ���Щ����������Ϳ��ԣ�������ʱ��û�����ͻ���ֻ��Ĳ��Ե���������������ͱ����ȶ��ԣ�
	cvFindContours(gray, storage, &contour, sizeof(CvContour),CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	cvDrawContours(gray, contour, CV_RGB(0,0,255), CV_RGB(0,0,255), 2,2,8, cvPoint(0,0)); 
	int area,maxArea = 10;//��������ֵ����10Pixel
	for(;contour;contour = contour->h_next)  
		{  
			area = fabs(cvContourArea( contour, CV_WHOLE_SEQ )); //��ȡ��ǰ�������  
			if(area > maxArea && area<((imgl->width)*(imgl->height)/1.5))  
			{  
				
				contmax = contour;  
				maxArea = area;  
			}  
		  }  
	aRect = cvBoundingRect(contmax, 0 );  
	//drawRect( imgl,aRect);

	//�Եڶ���ͼ�Ĵ���
	IplImage* imgr = cvLoadImage("E:\\test3.jpg");
	IplImage *gray2 = cvCreateImage(cvGetSize(imgr), IPL_DEPTH_8U, 1);
	cvCvtColor(imgr, gray2, CV_RGB2GRAY);
	cvSmooth(gray2,gray2, CV_GAUSSIAN ,5,5);
	cvNormalize(gray2,gray2, 1, 255, CV_MINMAX, NULL);
	cvAdaptiveThreshold(gray2,gray2,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,3,2);//��ȡ����Ӧ��ֵ�ķ���
	cvErode(gray2,gray2,0,2);
	cvDilate(gray2,gray2,0,2);
	cvFindContours(gray2, storage2, &contour2, sizeof(CvContour),CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	cvDrawContours(gray2, contour2, CV_RGB(0,0,255), CV_RGB(0,0,255), 2,2,8, cvPoint(0,0)); 
	area,maxArea = 10;//��������ֵ����10Pixel
	for(;contour2;contour2 = contour2->h_next)  
		{  
			area = fabs(cvContourArea( contour2, CV_WHOLE_SEQ )); //��ȡ��ǰ�������  
			if(area > maxArea && area<((imgl->width)*(imgl->height)/1.5))  
			{  
				
				contmax2 = contour2;  
				maxArea = area;  
			}  
		  }  
	
	aRect2 = cvBoundingRect(contmax2, 0 );  
	//drawRect( imgr,aRect2);

	//�ҵ�Ŀ���֮��Ĵ�����ROI��ʼƥ��
	//cvSetImageROI(imgl,cvRect(aRect.x,aRect.y,aRect.width,aRect.height));//t2ʱ�̵ĸ���Ȥ���򣬼�Ŀ������
	//cvSetImageROI(imgr,cvRect(aRect2.x,aRect2.y,aRect2.width,aRect2.height));//t2ʱ�̵ĸ���Ȥ���򣬼�Ŀ������

	surfDetDes(imgl,ipts1,true,3,3,5,0.0001f);// ��ȡt1ʱ����ͼ��������
		  for (int i = 0; i <ipts1.size(); i++)
      {
		   //matches[i].first.x= matches[i].first.x;
		   //matches[i].first.y= matches[i].first.y;
		   //matches[i].second.x= matches[i].second.x;
		   //matches[i].second.y= matches[i].second.y;
           drawPoint(imgl,ipts1[i]);
	 }

	surfDetDes(imgr,ipts2,true,3,3,5,0.0001f);// ��ȡt1ʱ����ͼ��������
			  for (int i = 0; i <ipts2.size(); i++)
      {
		   //matches[i].first.x= matches[i].first.x;
		   //matches[i].first.y= matches[i].first.y;
		   //matches[i].second.x= matches[i].second.x;
		   //matches[i].second.y= matches[i].second.y;
           drawPoint(imgr,ipts2[i]);
	 }
	//getMatches(ipts1,ipts2,matches);//ƥ��ǰ��ʱ�̵���ͼ��������
	//  for (int i = 0; i < matches.size(); i++)
 //     {
	//	   //matches[i].first.x= matches[i].first.x;
	//	   //matches[i].first.y= matches[i].first.y;
	//	   //matches[i].second.x= matches[i].second.x;
	//	   //matches[i].second.y= matches[i].second.y;
 //          drawPoint(imgl,matches[i].first);
 //          drawPoint(imgr,matches[i].second);
	// }
	//cvResetImageROI(imgl);
	//cvResetImageROI(imgr);
	//IplImage *stacked;
	//IplImage *stacked_ransac;
	//	 struct kd_node *kd_root;//k-d��������

	//stacked = stack_imgs( imgl, imgr );//�ϳ�ͼ����ʾ�������ֵ��ɸѡ���ƥ����  
 //   stacked_ransac = stack_imgs( imgl, imgr );//�ϳ�ͼ����ʾ��RANSAC�㷨ɸѡ���ƥ����  
 // //����ͼ2�������㼯feat2����k-d��������k-d������kd_root  
 //   kd_root = kdtree_build( ipts1,  );  
	////���������㼯feat1�����feat1��ÿ��������feat��ѡȡ���Ͼ����ֵ������ƥ��㣬�ŵ�feat��fwd_match����  
 //   for(int i = 0; i < n1; i++ )  
 //   {  
 //       feat = feat1+i;//��i���������ָ��  
 //       //��kd_root������Ŀ���feat��2������ڵ㣬�����nbrs�У�����ʵ���ҵ��Ľ��ڵ����  
 //       int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );  
 //       if( k == 2 )  
 //       {  
 //           d0 = descr_dist_sq( feat, nbrs[0] );//feat������ڵ�ľ����ƽ��  
 //           d1 = descr_dist_sq( feat, nbrs[1] );//feat��ν��ڵ�ľ����ƽ��  
 //           //��d0��d1�ı�ֵС����ֵNN_SQ_DIST_RATIO_THR������ܴ�ƥ�䣬�����޳�  
 //           if( d0 < d1 * NN_SQ_DIST_RATIO_THR )  
 //           {   //��Ŀ���feat������ڵ���Ϊƥ����  
	//			pt1.x= cvRound( feat->x );
	//			pt1.y= cvRound( feat->y );
	//			pt2.x=cvRound( nbrs[0]->x );
	//			pt2.y=cvRound( nbrs[0]->y );
 //               pt2.y += img1->height;//��������ͼ���������еģ�pt2�����������ͼ1�ĸ߶ȣ���Ϊ���ߵ��յ�  
 //               cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );//��������  
 //               matchNum++;//ͳ��ƥ���Եĸ���  
 //               feat1[i].fwd_match = nbrs[0];//ʹ��feat��fwd_match��ָ�����Ӧ��ƥ���  
 //           }  
 //       }  
 //       free( nbrs );//�ͷŽ�������  
 //   }  
	//cout<<"�������ֵ��ɸѡ���ƥ���Ը�����"<<matchNum<<endl;
	//cvNamedWindow(IMG_MATCH1);//��������  
 //   cvShowImage(IMG_MATCH1,stacked);//��ʾ  


 //   //����RANSAC�㷨ɸѡƥ���,����任����H  
	////FEATURE_BCK_MATCH
	////FEATURE_FWD_MATCH
	////FEATURE_MDL_MATCH
 //   CvMat * H = ransac_xform(feat1,n1,FEATURE_FWD_MATCH,lsq_homog,4,0.01,homog_xfer_err,3.0,&inliers,&n_inliers);  
 //   //������RANSAC�㷨ɸѡ��������㼯��inliers���ҵ�ÿ���������ƥ��㣬��������  
 //   for(int i=0; i<n_inliers; i++)  
 //   {  
 //       feat = inliers[i];//��i��������  
	//	pt1.x=cvRound(feat->x);
	//	pt1.y=cvRound(feat->y);
 //       pt2.x=cvRound(feat->fwd_match->x);
	//	pt2.y=cvRound(feat->fwd_match->y);
 //       pt2.y += img1->height;//��������ͼ���������еģ�pt2�����������ͼ1�ĸ߶ȣ���Ϊ���ߵ��յ�  
 //       cvLine(stacked_ransac,pt1,pt2,CV_RGB(255,0,255),1,8,0);//��������  
 //   }  
	//cout<<"��RANSAC�㷨ɸѡ���ƥ���Ը�����"<<n_inliers<<endl;
 //   cvNamedWindow(IMG_MATCH2);//��������  
 //   cvShowImage(IMG_MATCH2,stacked_ransac);//��ʾ 

 // cvReleaseImage( &stacked );
 // cvReleaseImage( &img1 );
 // cvReleaseImage( &img2 );
 // kdtree_release( kd_root );
 // free( feat1 );
 // free( feat2 );


	mixshow0(imgl,imgr);
	cvWaitKey();
	cvReleaseImage(&gray2);
	cvReleaseImage(&gray);
	cvReleaseImage(&imgr);
	cvReleaseImage(&imgl);
	
	cvDestroyAllWindows();
	
	return 0;
}

	
//	IplImage *ISec;
//	IplImage *IFir;
//	 CvMat* vdisp = cvCreateMat(cvGetSize(imgl).height,cvGetSize(imgl).width, CV_8U );
//	 CvMat* vdispB = cvCreateMat(cvGetSize(imgl).height,cvGetSize(imgl).width, CV_8U );
//	 cvNamedWindow("Fir",CV_WINDOW_AUTOSIZE);
//	 cvNamedWindow("FirGraySmoothHistDilate",CV_WINDOW_AUTOSIZE);
//	 cvNamedWindow("Fir2",CV_WINDOW_AUTOSIZE);
//	 cvNamedWindow("FirGraySmoothHistDilate2",CV_WINDOW_AUTOSIZE);
//	for(char i=0;i<2;i++){
//		if(i==0)
//		{
//	   cvCvtColor(imgl, gray, CV_RGB2GRAY);
//	 //sum_rgb(imgl,gray);
//	  //cvEqualizeHist(gray,gray);
//	cvSmooth(gray,gray,CV_GAUSSIAN,5,5);
//	 cvNormalize(gray, vdispB, 1, 255, CV_MINMAX, NULL);	//ͼ���һ��	
//		}
//		else
//		{
//		cvCvtColor(imgr, gray2, CV_RGB2GRAY);
//	//sum_rgb(imgr,gray2);
//	cvSmooth(gray2,gray2,CV_GAUSSIAN,5,5);
//	cvNormalize(gray2, vdispB, 1, 255, CV_MINMAX, NULL);	//ͼ���һ��	
//		}
//	cvAdaptiveThreshold(vdispB,vdisp,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,3,2);//��ȡ����Ӧ��ֵ�ķ���
//	cvErode(vdisp,vdisp,0,2);//��ʴ����
//	cvDilate(vdisp,vdisp,0,2);
//	//��ֵ����
//	if(i==0)
//	cvShowImage("FirGraySmoothHistN",vdisp);
//	//����
////cvErode(vdisp,vdisp,0,3);
////	cvDilate(vdisp,vdisp,0,3);
//	
//	cvFindContours( vdisp, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 
//	cvDrawContours(vdisp, contour, CV_RGB(0,0,255), CV_RGB(0,0,255), 2, 2, 8, cvPoint(0,0)); 
//	if(i==0){
//	cvShowImage("FirGraySmoothHistDilate",vdisp);
//	}else
//	{
//	cvShowImage("FirGraySmoothHistDilate2",vdisp);
//	}
//	int area,maxArea = 10;//��������ֵ����10Pixel
//	CvSeq *contmax = 0;
//	for(;contour;contour = contour->h_next)  
//		{  
//			area = fabs(cvContourArea( contour, CV_WHOLE_SEQ )); //��ȡ��ǰ�������  
//			//printf("area == %lf\n", area);  
//			if(area > maxArea )  
//			{  
//				contmax = contour;  
//				maxArea = area;  
//			}  
//		  }  
//	if(i==0){
//	 aRect = cvBoundingRect(contmax, 0 );  
//	}else
//	{
//	aRect1 = cvBoundingRect(contmax, 0 ); 
//	}
//	}
//	IFir= cvCreateImage(cvSize(aRect.width,aRect.height), imgl->depth,3);
//	//�ֱ����ø���Ȥ����
//	cvSetImageROI(imgl,cvRect(aRect.x,aRect.y,aRect.width,aRect.height));//t2ʱ�̵ĸ���Ȥ���򣬼�Ŀ������
//	cvSetImageROI(imgr,cvRect(aRect1.x,aRect1.y,aRect1.width,aRect1.height));//t2ʱ�̵ĸ���Ȥ���򣬼�Ŀ������
//
//	surfDetDes(imgl,ipts1,true,3,3,5,0.0001f);// ��ȡt1ʱ����ͼ��������
//	surfDetDes(imgr,ipts2,true,3,3,5,0.0001f);// ��ȡt1ʱ����ͼ��������
//	 getMatches(ipts1,ipts2,matches);//ƥ��ǰ��ʱ�̵���ͼ��������
//	  for (unsigned int i = 0; i < matches.size(); i++)
//      {
//		   matches[i].first.x= matches[i].first.x;
//		   matches[i].first.y= matches[i].first.y;
//		   matches[i].second.x= matches[i].second.x;
//		   matches[i].second.y= matches[i].second.y;
//           drawPoint(imgl,matches[i].first);
//           drawPoint(imgr,matches[i].second);
//	 }
//	 drawRect(imgl,aRect);
//	 drawRect(imgr,aRect1);
//	 cvShowImage("Fir",imgl);
//	 cvShowImage("Fir2",imgr);
//
//	cvWaitKey(0);


