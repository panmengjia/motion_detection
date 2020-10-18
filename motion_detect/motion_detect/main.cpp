
#include <opencv2/opencv.hpp>
#include <iostream>


using namespace std;
using namespace cv;


int main()
{
	RNG g_rng(12345);

	VideoCapture *cap=new VideoCapture(1, CAP_DSHOW);
	if (!cap->isOpened())
	{
		cout << "video is empty!" << endl;
		return -1;
	}
	Mat frame;

	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2(250,20,true);
	Mat fgmask, fgmask_thresh;

	Mat kernel_open = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat kernel_close = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

	while (1)
	{
		if (!cap->read(frame)) {
			cout << "没有抓到一帧" << endl;
			cap->release();
			cap = new VideoCapture(1, CAP_DSHOW);
			continue;
		}
		else
		{
			cap->read(frame);
			imshow("src", frame);
			waitKey(5);
			bg_model->apply(frame, fgmask);
			imshow("fgmask", fgmask);
			GaussianBlur(fgmask, fgmask, Size(5, 5), 0, 0);
			morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel_open);
			imshow("after morph_open", fgmask);
			morphologyEx(fgmask, fgmask, MORPH_CLOSE, kernel_close, Point(-1, -1), 2);
			imshow("after morph_close", fgmask);
			double otsu_thresh = threshold(fgmask, fgmask_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
			Mat labelImg(frame.size(), CV_32S);
			int nLabels = connectedComponents(fgmask_thresh, labelImg, 8);
			std::vector<Vec3b> colors(nLabels);
			colors[0] = Vec3b(0, 0, 0);
			for (int i = 1; i < nLabels; i++)
			{
				colors[i] = Vec3b(rand() & 255, rand() & 255, rand() & 255);
			}
			Mat fgmask_lc = Mat::zeros(frame.size(), CV_8UC3);
			for (int i = 0; i < frame.rows; i++)
			{
				for (int j = 0; j < frame.cols; j++)
				{
					int label = labelImg.at<int>(i, j);
					Vec3b& piexl = fgmask_lc.at<Vec3b>(i, j);
					piexl = colors[label];

				}
			}

			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(fgmask_thresh, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

			vector<vector<Point>> hull(contours.size());
			for (unsigned int i = 0; i < contours.size(); i++)
			{
				convexHull(Mat(contours[i]), hull[i], false);
			}

			Mat drawing_hull = Mat::zeros(frame.size(), CV_8UC3);
			Mat drawing_contours = Mat::zeros(frame.size(), CV_8UC3);
			vector<unsigned int> contour_area(contours.size());
			vector<unsigned int> hull_area(contours.size());
			for (unsigned int i = 0; i < contours.size(); i++)
			{
				contour_area[i] = contourArea(contours[i], false);
				hull_area[i] = contourArea(hull[i], false);
				Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
				drawContours(drawing_contours, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());//绘出轮廓
				drawContours(drawing_hull, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());//绘出轮廓的凸包
			}

			unsigned int contour_area_max = 0;
			unsigned int hull_area_max = 0;
			unsigned int contour_area_min = contour_area[0];
			unsigned int hull_area_min = hull_area[0];
			Point_<unsigned int> contour_max_area_index;// area index
			Point_<unsigned int> contour_min_area_index;// area index
			Point_<unsigned int> hull_max_area_index;// area index
			Point_<unsigned int> hull_min_area_index;// area index
			for (unsigned int i = 0; i < contours.size(); i++)
			{
				//contour_area_max = max(contour_area_max, contour_area[i]);
				//contour_area_min = min(contour_area_min, contour_area[i]);
				if (contour_area_max < contour_area[i])
				{
					contour_max_area_index = Point_<unsigned int>(contour_area[i], i);
					contour_area_max = contour_area[i];
				}
				if (contour_area_min > contour_area[i])
				{
					contour_min_area_index = Point_<unsigned int>(contour_area[i], i);
					contour_area_min = contour_area[i];
				}

				if (hull_area_max < hull_area[i])
				{
					hull_max_area_index = Point_<unsigned int>(hull_area[i], i);
					hull_area_max = hull_area[i];
				}
				if (hull_area_min > hull_area[i])
				{
					hull_min_area_index = Point_<unsigned int>(hull_area[i], i);
					hull_area_min = hull_area[i];
				}
			}

			//验证轮廓最大是否与凸包最大相当
			vector<vector<Point>> hull_max;
			hull_max.resize(1);
			//cv::InputArrayOfArrays hull_max_input = hull_max;
			convexHull(Mat(contours[contour_max_area_index.y]), hull_max[0], false);
			drawContours(frame, hull_max, 0, Scalar(255, 0, 0), 2, 8, vector<Vec4i>(), 0, Point());
			drawContours(frame, hull, contour_max_area_index.y, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());

			imshow("drawing_hull", drawing_hull);
			imshow("drawing_contours", drawing_contours);

			imshow("fgmask threshold", fgmask_thresh);
			imshow("fgmask_lc", fgmask_lc);

			imshow("frame", frame);
			waitKey(15);
			//static int i = 0;
			//cout << "i:	" << i++ << endl;
		}
	}
	return 0;
}


