
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stack>
#include <set>

using namespace cv;
using namespace std;

const double rate = 0.018868;
struct Data
{
	int minR, minC, maxR, maxC, length, centerR, centerC, order;
	Data() {}
	Data(int minr, int minc, int maxr, int maxc, int or )
	{
		minR = minr;
		minC = minc;
		maxR = maxr;
		maxC = maxc;
		order = or ;
		length = ((maxc - minc) + (maxr - minr)) / 2;
		centerR = (minr + maxr) / 2;
		centerC = (minc + maxc) / 2;
	}
	void toString()
	{
		printf("%.2lf\n", length * rate);
	}
	vector<pair<int, int>> Points; // first 为 row, second 为 col
};

int lc = 0, lr = 0, rc = 1280, rr = 1024, cnt_circle = 0;
double dia1 = 0, dia2 = 0, dia3 = 0, dia4 = 0;
double all_devi1 = 0, all_devi2 = 0, all_devi3 = 0, all_devi4 = 0;
Mat src;
string dst_name;
int dc[8] = { 0,1,1,1,0,-1,-1,-1 }, dr[8] = { -1,-1,0,1,1,1,0,-1 };
const double eps = 0.1;
bool used[2000][2000];
stack<pair<int, int>> sta;
vector<Data> result, make_circle;

void init(); //初始化数据
void make_name(int i); //产生图片的名字
bool cmp(Data d1, Data d2); //对数据排序时的比较函数
void create_contours(); //画轮廓图
bool isStore(int minC, int minR, int maxC, int  maxR); //判断找到的圆是否合格
void find_adjacent(); //提取连通域
void get_circle(); //
void fit_circle(int k); //使用最小二乘法进行圆拟合
void draw_circle(); //画圆
double get_devi(double A, double B, double R, int k);

int main(int argc, char** argv)
{
	freopen("output_circle_size.txt", "w", stdout);
	printf("注:单位 厘米 , R 半径\n\n");
	for (int k = 1; k< argc; k++)
	{
		init();
		make_name(k);
		src = imread(argv[k], 1);
		create_contours();
		find_adjacent();
		get_circle();
		printf("样例%d\n", k);
		draw_circle();
		printf("\n");
		dst_name.clear();
	}
	printf("平均数据:\n");
	printf("灯孔1R:%.2lf", dia1 * rate / 4);
	printf("  偏差值:%.0lf\n", all_devi1 / 4);
	printf("灯孔2R:%.2lf", dia2 * rate / 4);
	printf("  偏差值:%.0lf\n", all_devi2 / 4);
	printf("灯柱R:%.2lf", dia3 * rate / 4);
	printf("  偏差值:%.0lf\n", all_devi3 / 4);
	printf("外圈R:%.2lf", dia4 * rate / 4);
	printf("  偏差值:%.0lf\n", all_devi4 / 4);
	fclose(stdout);
	return(0);
}

bool cmp(Data d1, Data d2)
{
	return d1.length < d2.length;
}

void init()
{
	cnt_circle = 0;
	result.clear();
	make_circle.clear();
	while (!sta.empty()) sta.pop();
}

void make_name(int i)
{
	string tmpStr;
	stringstream ss;
	ss << i;
	ss >> tmpStr;
	dst_name += "result";
	dst_name += tmpStr;
	dst_name += ".jpg";
}

void create_contours()
{
	Mat src_gray, canny_output;
	int thresh = 10, max_thresh = 255;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	Canny(src_gray, canny_output, thresh, thresh * 1.5, 3);

	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	//rectangle(src, Point(350,200), Point(900,720), Scalar(255, 255, 255), 2, 8, 0);
	//imshow("sfsd", src);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);//定义颜色为白色
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
	}
	imwrite("canny.jpg", drawing);
	//imshow("canny",drawing);
	waitKey(0);
}

bool isStore(int minC, int minR, int maxC, int  maxR)
{
	int CL = maxC - minC, RL = maxR - minR;
	double rate = CL * 1.0 / RL;
	if (fabs(rate - 1) <= eps && RL >= 20 && CL >= 20)
	{
		return true;
	}
	return false;
}

void find_adjacent()
{
	memset(used, false, sizeof used);
	Mat S, mid, dst;
	S = imread("canny.jpg");
	cvtColor(S, mid, CV_BGR2GRAY);
	threshold(mid, dst, 145, 255, THRESH_BINARY);

	vector<pair<int, int>> tmpPoint;

	for (int r = lr; r < rr; r++)
	{
		for (int c = lc; c < rc; c++)
		{
			tmpPoint.clear();
			if (!used[r][c] && dst.at<uchar>(r, c) == 255)
			{
				int minC = c, minR = r, maxC = c, maxR = r;
				sta.push(make_pair(r, c));
				used[r][c] = true;
				tmpPoint.push_back(make_pair(r, c));
				while (!sta.empty())
				{
					int tmpR = sta.top().first, tmpC = sta.top().second;
					sta.pop();
					for (int k = 0; k <8; k++)
					{
						int tc = tmpC + dc[k], tr = tmpR + dr[k];
						if (tr < rr && tr >= lr && tc < rc&& tc >= lc)
						{
							if (!used[tr][tc] && dst.at<uchar>(tr, tc) == 255)
							{
								sta.push(make_pair(tr, tc));
								tmpPoint.push_back(make_pair(tr, tc));
								used[tr][tc] = true;
								minC = min(minC, tc);
								minR = min(minR, tr);
								maxC = max(maxC, tc);
								maxR = max(maxR, tr);
							}
						}
					}
				}
				if (isStore(minC, minR, maxC, maxR))
				{
					Data d = Data(minR, minC, maxR, maxC, cnt_circle++);
					d.Points = tmpPoint;
					result.push_back(d);
					tmpPoint.clear();
				}
			}
		}
	}
}

void get_circle()
{
	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	int minr, minc, maxr, maxc;
	minr = result[0].minR;
	minc = result[0].minC;
	maxr = result[0].maxR;
	maxc = result[0].maxC;
	for (int i = 1; i < result.size(); i++)
	{
		minr = min(minr, result[i].minR);
		minc = min(minc, result[i].minC);
		maxr = max(maxr, result[i].maxR);
		maxc = max(maxc, result[i].maxC);
	}
	int length = ((maxc - minc) + (maxr - minr)) / 2;
	int centerR = (maxr + minr) / 2;
	int centerC = (maxc + minc) / 2;

	for (int i = 0; i <result.size(); i++)
	{
		int tempC = result[i].centerC;
		int tempR = result[i].centerR;
		if (result[i].minC <= minc + 50 && result[i].minC != minc) continue;
		if (tempC >= centerC - 50 && tempC <= centerC + 50 && tempR <= centerR + 50 && tempR >= centerR - 50)
		{
			make_circle.push_back(result[i]);
		}
	}
	result.clear();
	sort(make_circle.begin(), make_circle.end(), cmp);

	Data d1 = make_circle[0];
	Data d2 = make_circle[1];
	//rectangle(src, Point(make_circle[0].minC, make_circle[0].minR), Point(make_circle[0].maxC, make_circle[0].maxR), Scalar(255, 255, 255), 1, 8, 0);
	//rectangle(src, Point(make_circle[1].minC, make_circle[1].minR), Point(make_circle[1].maxC, make_circle[1].maxR), Scalar(255, 255, 255), 1, 8, 0);
	result.push_back(d1);
	result.push_back(d2);
	//cout << d1.order << endl;
	//cout << d2.order << endl;
	/*for (int i = 0; i < 2; i++)
	{
	Point center(make_circle[i].centerC, make_circle[i].centerR);
	int radius = make_circle[i].length / 2;
	circle(src, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}*/

	//if (abs(d1.length - d2.length) > 2) cout << "灯孔大小不一致" << endl;
	int std_length = (d1.length + d2.length) / 2;
	for (int i = 2; i < make_circle.size(); i++)
	{
		if (make_circle[i].length > 2 * std_length)
		{
			//rectangle(src, Point(make_circle[i].minC, make_circle[i].minR), Point(make_circle[i].maxC, make_circle[i].maxR), Scalar(255, 255, 255), 1, 8, 0);
			/*Point center(make_circle[i].centerC, make_circle[i].centerR);
			int radius = make_circle[i].length / 2;
			circle(src, center, radius, Scalar(0, 0,255), 2, 8, 0);*/
			result.push_back(make_circle[i]);
			//cout << make_circle[i].order << endl;
		}
	}
	//imshow("window", src);
	//imwrite(dst_name, src);
	waitKey(0);
}

void fit_circle(int k) //最小二乘法来拟合圆
{
	double X1 = 0;
	double Y1 = 0;
	double X2 = 0;
	double Y2 = 0;
	double X3 = 0;
	double Y3 = 0;
	double X1Y1 = 0;
	double X1Y2 = 0;
	double X2Y1 = 0;

	for (int i = 0; i<result[k].Points.size(); i++)
	{
		X1 = X1 + result[k].Points[i].second;
		Y1 = Y1 + result[k].Points[i].first;
		X2 = X2 + result[k].Points[i].second*result[k].Points[i].second;
		Y2 = Y2 + result[k].Points[i].first*result[k].Points[i].first;
		X3 = X3 + result[k].Points[i].second*result[k].Points[i].second*result[k].Points[i].second;
		Y3 = Y3 + result[k].Points[i].first*result[k].Points[i].first*result[k].Points[i].first;
		X1Y1 = X1Y1 + result[k].Points[i].second*result[k].Points[i].first;
		X1Y2 = X1Y2 + result[k].Points[i].second*result[k].Points[i].first*result[k].Points[i].first;
		X2Y1 = X2Y1 + result[k].Points[i].second*result[k].Points[i].second*result[k].Points[i].first;
	}

	double C, D, E, G, H, N;
	double a, b, c;
	N = result[k].Points.size();
	C = N*X2 - X1*X1;
	D = N*X1Y1 - X1*Y1;
	E = N*X3 + N*X1Y2 - (X2 + Y2)*X1;
	G = N*Y2 - Y1*Y1;
	H = N*X2Y1 + N*Y3 - (X2 + Y2)*Y1;
	a = (H*D - E*G) / (C*G - D*D);
	b = (H*C - E*D) / (D*D - G*C);
	c = -(a*X1 + b*Y1 + X2 + Y2) / N;

	double A, B, R;
	A = a / (-2);
	B = b / (-2);
	R = sqrt(a*a + b*b - 4 * c) / 2;

	Point center(A, B);
	circle(src, center, R, Scalar(0, 0, 255), 2, 8, 0);
	imwrite(dst_name, src);
	double devi = get_devi(A, B, R, k);
	devi /= N;
	if (k == 0)
	{
		printf("灯孔1R:%.2lf", R * rate);
		printf("  偏差值:%.0lf\n", devi);
		dia1 += R;
		all_devi1 += devi;
	}
	if (k == 1)
	{
		printf("灯孔2R:%.2lf", R  * rate);
		printf("  偏差值:%.0lf\n", devi);
		dia2 += R;
		all_devi2 += devi;
	}
	if (k == 2)
	{
		printf("灯柱R:%.2lf", R * rate);
		printf("  偏差值:%.0lf\n", devi);
		dia3 += R;
		all_devi3 += devi;
	}
	if (k == 3)
	{
		printf("外圈R:%.2lf", R * rate);
		printf("  偏差值:%.0lf\n", devi);
		dia4 += R;
		all_devi4 += devi;
	}
}

void draw_circle()
{
	for (int i = 0; i < result.size(); i++)
	{
		fit_circle(i);
	}
	//imshow("window", src);
	waitKey(0);
}

double get_devi(double X, double Y, double R, int k)
{
	double res = 0, R2 = pow(R, 2);
	for (int i = 0; i < result[k].Points.size(); i++)
	{
		double x = result[k].Points[i].second;
		double y = result[k].Points[i].first;
		double d2 = pow(x - X, 2) + pow(y - Y, 2);
		res += pow(d2 - R2, 2);
	}
	return res;
}


