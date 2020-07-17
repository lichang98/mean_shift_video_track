// RGB format mean shift tracking
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <unistd.h>

using namespace std;
using namespace cv;

const static int HIST_BIN_NUM = 16; // for each color channel, the number of histgram bins
const static double EPSILON = 0.000001;
const static int MAX_ITER = 10;
// object may be temporarily occulude by other objects, in this case, the new position calculated may be unnormal
const static int EXCEPTION_THRESHOLD=20; 

template <class T>
struct Dim3Vec
{
    vector<vector<vector<T>>> data;
    int sizes[3];

    Dim3Vec(int dims[])
    {
        sizes[0] = dims[0];
        sizes[1] = dims[1];
        sizes[2] = dims[2];
        data = vector<vector<vector<T>>>(dims[0], vector<vector<T>>(dims[1], vector<T>(dims[2], 0)));
    }

    void operator=(const Dim3Vec<T> &other)
    {
        for (int i = 0; i < sizes[0]; ++i)
            for (int j = 0; j < sizes[1]; ++j)
                for (int k = 0; k < sizes[2]; ++k)
                    this->data[i][j][k] = other.data[i][j][k];
    }

    T get(int i0, int i1, int i2)
    {
        return data[i0][i1][i2];
    }

    void set(int i0, int i1, int i2, T val)
    {
        data[i0][i1][i2] = val;
    }

    void inc(int i0, int i1, int i2, T val)
    {
        data[i0][i1][i2] += val;
    }

    T sum()
    {
        T val = 0;
        for (int i = 0; i < sizes[0]; ++i)
            for (int j = 0; j < sizes[1]; ++j)
                for (int k = 0; k < sizes[2]; ++k)
                    val += data[i][j][k];
        return val;
    }

    void convert_to(double alpha = 1.0, double beta = 0.0)
    {
        for (int i = 0; i < sizes[0]; ++i)
            for (int j = 0; j < sizes[1]; ++j)
                for (int k = 0; k < sizes[2]; ++k)
                    data[i][j][k] = data[i][j][k] * alpha + beta;
    }

    void div(Dim3Vec &other)
    {
        for (int i = 0; i < sizes[0]; ++i)
            for (int j = 0; j < sizes[1]; ++j)
                for (int k = 0; k < sizes[2]; ++k)
                    data[i][j][k] = data[i][j][k] * 1.0 / (other.get(i, j, k) + EPSILON);
    }

    void mul(Dim3Vec &other)
    {
        for (int i = 0; i < sizes[0]; ++i)
            for (int j = 0; j < sizes[1]; ++j)
                for (int k = 0; k < sizes[2]; ++k)
                    data[i][j][k] *= other.get(i, j, k);
    }

    void show()
    {
        for (int i = 0; i < sizes[0]; ++i)
            for (int j = 0; j < sizes[1]; ++j)
                for (int k = 0; k < sizes[2]; ++k)
                    printf("%lf\t", data[i][j][k]);
        printf("\n");
    }
};

struct pos
{
    int row, col;
    pos(int _row, int _col) : row(_row), col(_col) {}
    pos() : row(0), col(0) {}

    bool near_same(pos &other, int epsilon = 2.0)
    {
        if (sqrt(pow(row - other.row, 2) + pow(col - other.col, 2)) < epsilon)
            return true;
        else
            return false;
    }

    void operator=(const pos &other)
    {
        this->row = other.row;
        this->col = other.col;
    }

    bool isfar(pos &other)
    {
        if(sqrt(pow(row-other.row,2) + pow(col-other.col,2)) > EXCEPTION_THRESHOLD)
            return true;
        else
            return false;
    }
};

Dim3Vec<double> subwin_color_p(Mat region, Mat kernel_weights)
{
    int row = region.rows, col = region.cols;
    int size[] = {HIST_BIN_NUM, HIST_BIN_NUM, HIST_BIN_NUM};
    Dim3Vec<double> color_p(size);

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            int b = (int)region.at<Vec3b>(i, j)[0], g = (int)region.at<Vec3b>(i, j)[1], r = (int)region.at<Vec3b>(i, j)[2];
            color_p.inc(b * HIST_BIN_NUM / 256, g * HIST_BIN_NUM / 256, r * HIST_BIN_NUM / 256, kernel_weights.at<double>(i, j));
        }
    }
    double norm = color_p.sum();
    color_p.convert_to(1.0 / norm);
    return color_p;
}

double weight(Dim3Vec<double> color_p, Dim3Vec<double> mode_q, int b, int g, int r)
{
    return sqrt(mode_q.get(b * HIST_BIN_NUM / 256, g * HIST_BIN_NUM / 256, r * HIST_BIN_NUM / 256) / (EPSILON + color_p.get(b * HIST_BIN_NUM / 256, g * HIST_BIN_NUM / 256, r * HIST_BIN_NUM / 256)));
}

double bhat_coef(Dim3Vec<double> color_p, Dim3Vec<double> mode_q)
{
    double res = 0;
    mode_q.mul(color_p);

    for (int i = 0; i < mode_q.sizes[0]; ++i)
        for (int j = 0; j < mode_q.sizes[1]; ++j)
            for (int k = 0; k < mode_q.sizes[2]; ++k)
                res += sqrt(mode_q.get(i, j, k));
    return res;
}

pos calc_pos(Mat img, pos prev, int region_roww, int region_colw, Mat kernel_weights, Dim3Vec<double> mode_q, Dim3Vec<double> color_p)
{
    pos ans;
    double de = 0, ne1 = 0, ne2 = 0;

    for (int i = prev.row; i < prev.row + region_roww; ++i)
    {
        for (int j = prev.col; j < prev.col + region_colw; ++j)
        {
            double w = weight(color_p, mode_q, (int)img.at<Vec3b>(i, j)[0], (int)img.at<Vec3b>(i, j)[1], (int)img.at<Vec3b>(i, j)[2]);
            de += w * kernel_weights.at<double>(i,j);
            ne1 += i * w * kernel_weights.at<double>(i,j);
            ne2 += j * w * kernel_weights.at<double>(i,j);
        }
    }
    ans.row = (int)(ne1 / de) - region_roww / 2;
    ans.col = (int)(ne2 / de) - region_colw / 2;
    return ans;
}

pos mean_shift_update(Mat img, pos prev, int region_roww, int region_colw, Dim3Vec<double> mode_q)
{
    Mat kernel_weights = getGaussianKernel(region_roww, 1) * getGaussianKernel(region_colw, 1).t();

    for (int epoch = 0; epoch < MAX_ITER; ++epoch)
    {
        double prev_coef = bhat_coef(subwin_color_p(Mat(img, Rect(prev.col, prev.row, region_colw, region_roww)), kernel_weights), mode_q);
        pos nw_pos = calc_pos(img, prev, region_roww, region_colw, kernel_weights, mode_q, subwin_color_p(Mat(img, Rect(prev.col, prev.row, region_colw, region_roww)), kernel_weights));
        
        // In the case the occulusion occured, the calculated new position may be very incorrect
        if(nw_pos.isfar(prev))
        {
            return prev;
        }
        printf("\t first get pos = (%d,%d)\n",nw_pos.row,nw_pos.col);
        double nw_coef = bhat_coef(subwin_color_p(Mat(img, Rect(nw_pos.col, nw_pos.row, region_colw, region_roww)), kernel_weights), mode_q);
        printf("\t iteration, new pos=(%d,%d) coef=%lf, prev_pos=(%d,%d), coef=%lf\n", nw_pos.row, nw_pos.col, nw_coef, prev.row, prev.col, prev_coef);
        bool flag = true;

        while (nw_coef < prev_coef)
        {
            nw_pos.row = (int)round((nw_pos.row + prev.row) / 2.0);
            nw_pos.col = (int)round((nw_pos.col + prev.col) / 2.0);
            if (nw_pos.near_same(prev))
            {
                flag = false;
                break;
            }
            nw_coef = bhat_coef(subwin_color_p(Mat(img, Rect(nw_pos.col, nw_pos.row, region_colw, region_roww)), kernel_weights), mode_q);
        }
        if (!flag)
        {
            return prev;
        }
        else
        {
            prev = nw_pos;
        }
    }
    return prev;
}

void draw_bound(Mat &img, pos start, int region_roww, int region_colw)
{
    rectangle(img, Rect(start.col, start.row, region_colw, region_roww), Scalar(0, 255, 0), 1, LINE_8, 0);
}

int main(int argc, char const *argv[])
{
    pos center(240, 300);
    int region_roww = 40, region_colw = 30;
    char *video_name = (char *)"mot.avi";
    Mat gaussian_kernel = getGaussianKernel(region_roww, 1.0) * getGaussianKernel(region_colw, 1.0).t();
    Mat frame;
    VideoCapture capture(video_name);
    capture >> frame;

    // draw_bound(frame,center,region_roww,region_colw);
    // imshow("",frame);
    // waitKey();

    Dim3Vec<double> mode_q = subwin_color_p(Mat(frame, Rect(center.col, center.row, region_colw, region_roww)), gaussian_kernel);

    namedWindow("");

    while (!frame.empty())
    {
        pos nw_pos = mean_shift_update(frame, center, region_roww, region_colw, mode_q);
        printf("new frmae, pos=(%d,%d)\n", nw_pos.row, nw_pos.col);
        draw_bound(frame, nw_pos, region_roww, region_colw);
        imshow("", frame);
        center = nw_pos;
        capture >> frame;
        waitKey(100);
    }
    return 0;
}
