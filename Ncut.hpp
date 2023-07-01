#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <set>
#include <map>

/**
 * @brief image sigmentation with normalized cuts and image segmentation method
 * @details this method is a c++ implementation of paper [1] and [2]
 * [1] J. Shi and J. Malik, “Normalized cuts and image segmentation,” 
 * in Proceedings of IEEE Computer Society Conference on Computer Vision 
 * and Pattern Recognition, Jun. 1997, pp. 731–737. doi: 10.1109/CVPR.1997.609407.
 * 
 * [2] J. Shi and J. Malik, “Normalized cuts and image segmentation,” 
 * IEEE Transactions on Pattern Analysis and Machine Intelligence,
 *  vol. 22, no. 8, pp. 888–905, Aug. 2000, doi: 10.1109/34.868688.
 */
class Ncut
{
public:

/**
 * @brief vertex of graph
 * 
 */
	struct Vertex_t
	{
		int i = 0; //i,j is the vertex in original image
		int j = 0;
		int index=0; //index is the index of vertex in graph
		cv::Vec3b color; //color in original pixel
		std::vector<int> neighbors; //neighbors of vertex
		std::vector<float> weights; //connected weights of neighbors

	};
	using Graph_t = std::vector<Vertex_t>; //define graph as a vector of vertex
	const double PI=3.1415926; //pi
public:

/**
 * @brief Construct a new Ncut object
 * 
 * @param r neighborhood radius
 * @param sigmaI parameter for color similarity
 * @param sigmaX parameter for spatial similarity
 * @param k cluster number, this number must be 2^k
 */
	Ncut(int r, double sigmaI, double sigmaX, int k = 4);
	/**
	 * @brief Destroy the Ncut object
	 * 
	 */
	~Ncut();

	/**
	 * @brief operator() perform normalized cuts and image segmentation
	 * 
	 * @param image image input
	 * @param mask segmentation output
	 */
	void operator()(const cv::Mat& image, cv::Mat& mask);


private:
	const int r;
	const double sigmaI;
	const double sigmaX;
	const int k;

private:
/**
 * @brief Build graph for normalized cuts
 * 
 * @param image image input
 * @param graph graph output

 */
	void BuildGraph(const cv::Mat& image, Graph_t& graph);

	/**
	 * @brief calculate D matrix
	 * 
	 * @param graph graph input
	 * @return cv::Mat D matrix
	 */
	cv::Mat D(const Graph_t& graph);

	/**
	 * @brief calculate W matrix
	 * 
	 * @param graph graph input
	 * @return cv::Mat W matrix
	 */
	cv::Mat W(const Graph_t& graph);

	/**
	 * @brief solve eigen problem (D-W)y=\lambda Dy
	 * 
	 * @param D D matrix
	 * @param W W matrix
	 * @return cv::Mat second smallest eigen vector
	 */
	cv::Mat SolveEigen(const cv::Mat& D, const cv::Mat& W);

	/**
	 * @brief calculate connection weight between two vertex
	 * 
	 * @param i color of vertex i
	 * @param j color of vertex j
	 * @param posi position of vertex i
	 * @param posj position of vertex j
	 * @return double weight
	 */
	double Omega(cv::Vec3b& i, cv::Vec3b& j,cv::Vec2i posi,cv::Vec2i posj);
	
	/**
	 * @brief sigment graph into two parts
	 * 
	 * @param U second smallest eigen vector from SolveEigen
	 * @param graph graph input
	 * @param P subgraph 1
	 * @param N subgraph 2
	 */
	void Sigment(const cv::Mat& U, const Graph_t& graph, Graph_t& P, Graph_t& N);

	/**
	 * @brief reshape graph into image
	 * 
	 * @param mask segmentation result
	 * @param G graph
	 * @param ID clustering ID
	 */
	void Merge(cv::Mat& mask, const Graph_t& G, int ID);

	/**
	 * @brief segmentation recursion to segment multiple clusters
	 * 
	 * @param Graph Graph input
	 * @param mask segmentation result
	 * @param ctl recursion control
	 * @param ColorRange color control for better visualization
	 */
	void Recursion(const Graph_t& Graph, cv::Mat& mask, int ctl = 0, cv::Vec2b ColorRange = cv::Vec2b(0, 255));
};

Ncut::Ncut(int r, double sigmaI, double sigmaX,int k)
	: r(r), 
	sigmaI(sigmaI), 
	sigmaX(sigmaX),
	k(std::log2(k)) //k must be 2^k, for recursion
{
}

Ncut::~Ncut()
{
}

void Ncut::operator()(const cv::Mat& image, cv::Mat& mask)
{
	cv::Mat HSV;
	cv::cvtColor(image, HSV, cv::COLOR_BGR2HSV); //convert to HSV color space
	//OpenCV HSV is different from standard definition, H is 0-180, S is 0-255, V is 0-255
	Graph_t graph;
	BuildGraph(HSV, graph);//build graph
	mask = cv::Mat::zeros(image.size(), CV_8UC1);//create mask
	this->Recursion(graph, mask, k, cv::Vec2b(0, 255)); //start recursive segmentation
}

void Ncut::BuildGraph(const cv::Mat& image, Graph_t& graph)
{
	graph.clear();
	graph.reserve(image.cols * image.rows);
	//build vertex
	for (size_t i = 0; i < image.rows; i++)
	{
		for (size_t j = 0; j < image.cols; j++)
		{
			Vertex_t vertex;
			vertex.i = i;
			vertex.j = j;
			vertex.index = i * image.cols + j;
			vertex.color = image.at<cv::Vec3b>(i, j);
			graph.push_back(vertex);
		}
	}


	//build edges
	for (size_t i = 0; i < image.rows; i++)
	{
		for (size_t j = 0; j < image.cols; j++)
		{
			for (size_t k = 0; k < 2*r+2; k++)
			{
				for (size_t l = 0; l < 2*r+2; l++)
				{
					int x_ = i - r + k;
					int y_ = j - r + l;
					if(x_==i&&y_==j)
						continue;
					if (x_ < 0 || x_ >= image.rows || y_ < 0 || y_ >= image.cols)
						continue;

					cv::Vec3b colorj = image.at<cv::Vec3b>(x_, y_);
					cv::Vec3b colori = image.at<cv::Vec3b>(i, j);
					double w = Omega(colori, colorj, cv::Vec2i(i, j), cv::Vec2i(x_, y_)); //calculate weight
					if (w != 0)
					{
						int indexi=i*image.cols+j;
						int indexj=x_*image.cols+y_;
						graph[indexi].neighbors.push_back(indexj);
						graph[indexi].weights.push_back(w);
					}
				}
			}
		}
	}
}

cv::Mat Ncut::D(const Graph_t& graph)
{
	cv::Mat DMat=cv::Mat::zeros(graph.size(),graph.size(),CV_64FC1);
	for (size_t i = 0; i < graph.size(); i++)
	{
		double sum = 0;
		for (size_t j = 0; j < graph[i].neighbors.size(); j++)
		{
			sum += graph[i].weights[j];
		}
		DMat.at<double>(i, i) = sum;
	}
	return DMat;
}

cv::Mat Ncut::W(const Graph_t& graph)
{
	cv::Mat WMat = cv::Mat::zeros(graph.size(), graph.size(), CV_64FC1);
	for (size_t i = 0; i < graph.size(); i++)
	{
		for (size_t j = 0; j < graph[i].neighbors.size(); j++)
		{
			WMat.at<double>(i, graph[i].neighbors[j]) = graph[i].weights[j];
		}
	}
	assert(!cv::countNonZero(WMat - WMat.t())); //check if W is symmetric
	return WMat;
}

cv::Mat Ncut::SolveEigen(const cv::Mat& D, const cv::Mat& W)
{
	//solve generalized eigenvalue problem
	cv::Mat invD = D.inv();
	cv::Mat invD2 = invD;
	cv::sqrt(invD, invD2);
	cv::Mat L= invD2*(D-W)*invD2;
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(L, eigenvalues, eigenvectors);
	cv::Mat U = invD2*eigenvectors;
	return U.row(U.rows - 2).clone();
}

inline double Ncut::Omega(cv::Vec3b& i, cv::Vec3b& j, cv::Vec2i posi, cv::Vec2i posj)
{
	if (cv::norm(posi - posj) > this->r)
		return 0;
	cv::Vec3d Fi;
	//OpenCV's HSV definition is different from standard definition, have to convert to standard hsv
	Fi(0) = i(2)/255.0;
	Fi(1) = (i(2) / 255.0) * (i(1) / 255.0) * sin((i(0) / 180.0) * 2 * PI);
	Fi(2) = (i(2) / 255.0) * (i(1) / 255.0) * cos((i(0) / 180.0) * 2 * PI);

	cv::Vec3d Fj;
	Fj(0) = j(2) / 255.0;
	Fj(1) = (j(2) / 255.0) * (j(1) / 255.0) * sin((j(0) / 180.0) * 2 * PI);
	Fj(2) = (j(2) / 255.0) * (j(1) / 255.0) * cos((j(0) / 180.0) * 2 * PI);

	double F1=std::exp(-cv::norm(Fi - Fj) / this->sigmaI);
	double F2=std::exp(-cv::norm(posi - posj) / this->sigmaX);
	CV_Assert((F1 * F2) != 0.0); //check if F1*F2 is zero, if so, the weight is zero
	return F1 * F2;
}


void Ncut::Sigment(const cv::Mat& U, const Graph_t& graph, Graph_t& P, Graph_t& N)
{
	P.clear();
	N.clear();
	std::map<int, int> PMap;
	std::map<int, int> NMap;
	int PIndex = 0;
	int NIndex = 0;

	//split graph into two subgraph
	for (size_t i = 0; i < U.cols; i++)
	{
		if (U.at<double>(i) > 0)
		{
			P.push_back(graph[i]);
			PMap.insert(std::make_pair(graph[i].index, PIndex++)); //record new index in subgraph
		}
		else
		{
			N.push_back(graph[i]);
			NMap.insert(std::make_pair(graph[i].index, NIndex++));
		}
	}

	//remove edge
	//remove edge in P
	for (size_t i = 0; i < P.size(); i++)
	{
		std::vector<int> NewNeighbors;
		std::vector<float> NewWeights;
		for (size_t j = 0; j < P[i].neighbors.size(); j++)
		{
			if (PMap.count(P[i].neighbors[j]) != 0)
			{
				NewNeighbors.push_back(PMap[P[i].neighbors[j]]);
				NewWeights.push_back(P[i].weights[j]);
			}
		}
		P[i].neighbors = NewNeighbors;
		P[i].weights = NewWeights;
	}

	//remove edge in N
	for (size_t i = 0; i < N.size(); i++)
	{
		std::vector<int> NewNeighbors;
		std::vector<float> NewWeights;
		for (size_t j = 0; j < N[i].neighbors.size(); j++)
		{
			if (NMap.count(N[i].neighbors[j]) != 0)
			{
				NewNeighbors.push_back(NMap[N[i].neighbors[j]]);
				NewWeights.push_back(N[i].weights[j]);
			}
		}
		N[i].neighbors = NewNeighbors;
		N[i].weights = NewWeights;
	}

}

void Ncut::Merge(cv::Mat& mask, const Graph_t& G, int ID)
{
	for (auto&& vertex : G)
	{
		mask.at<uchar>(vertex.i, vertex.j) = ID; //transform graph to mask image
	}
}

void Ncut::Recursion(const Graph_t& Graph, cv::Mat& mask, int ctl, cv::Vec2b ColorRange)
{
	//if no more recursion, merge the graph
	if (ctl == 0)
	{
		int mid = (ColorRange(0) + ColorRange(1)) / 2;
		Merge(mask, Graph, mid);
	}
	else
	{
		cv::Mat DMat = D(Graph);
		cv::Mat WMat = W(Graph);
		cv::Mat U = SolveEigen(DMat, WMat);
		Graph_t P, N;
		Sigment(U, Graph, P, N);
		int mid=(ColorRange(0)  +ColorRange(1))/2;
		Recursion(P,mask, ctl - 1, cv::Vec2b(mid+1, ColorRange(1)));
		Recursion(N, mask, ctl - 1, cv::Vec2b(ColorRange(0), mid));
	}
}

