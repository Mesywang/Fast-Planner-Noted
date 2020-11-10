#include "bspline_opt/bspline_optimizer.h"
#include <nlopt.hpp>
using namespace std;

namespace dyn_planner
{
void BsplineOptimizer::setControlPoints(Eigen::MatrixXd points)
{
    this->control_points_ = points;
    this->start_id_ = order_; // 为了保证收尾的边界条件, 不对前后各p个控制点进行优化
    this->end_id_ = this->control_points_.rows() - order_;
    use_guide_ = false;
}

void BsplineOptimizer::setOptimizationRange(int start, int end)
{
    this->start_id_ = min(max(start, order_), int(control_points_.rows() - order_));
    this->end_id_ = min(max(end, order_), int(control_points_.rows() - order_));
    cout << "opt range:" << this->start_id_ << ", " << this->end_id_ << endl;
}

void BsplineOptimizer::setParam(ros::NodeHandle &nh)
{
    nh.param("optimization/lamda1", lamda1_, -1.0);     // 10
    nh.param("optimization/lamda2", lamda2_, -1.0);     // 0.6
    nh.param("optimization/lamda3", lamda3_, -1.0);     // 0.00001
    nh.param("optimization/lamda4", lamda4_, -1.0);     // 0.001
    nh.param("optimization/lamda5", lamda5_, -1.0);
    nh.param("optimization/dist0", dist0_, -1.0);       // 0.8
    nh.param("optimization/dist1", dist1_, -1.0);
    nh.param("optimization/max_vel", max_vel_, -1.0);   // 2.0
    nh.param("optimization/max_acc", max_acc_, -1.0);   // 1.0
    nh.param("optimization/max_iteration_num", max_iteration_num_, -1); // 100
    nh.param("optimization/algorithm", algorithm_, -1); // 11 -> LBFGS(unconstrained)
    nh.param("optimization/order", order_, -1);

    std::cout << "lamda1: " << lamda1_ << std::endl;
    std::cout << "lamda2: " << lamda2_ << std::endl;
    std::cout << "lamda3: " << lamda3_ << std::endl;
    std::cout << "lamda4: " << lamda4_ << std::endl;
}

void BsplineOptimizer::setBSplineInterval(double ts)
{
    this->bspline_interval_ = ts;
}

void BsplineOptimizer::setEnvironment(const EDTEnvironment::Ptr &env)
{
    this->edt_env_ = env;
}

Eigen::MatrixXd BsplineOptimizer::getControlPoints()
{
    return this->control_points_;
}

/* best algorithm_ is 40: SLSQP(constrained), 11 LBFGS(unconstrained) */
void BsplineOptimizer::optimize(int end_cons, bool dynamic, double time_start)
{
    /* ---------- initialize solver ---------- */
    end_constrain_ = end_cons;
    dynamic_ = dynamic;
    time_traj_start_ = time_start;
    iter_num_ = 0;

    if (end_constrain_ == HARD_CONSTRAINT)
    {
        variable_num_ = 3 * (end_id_ - start_id_); // 需优化的控制点数量, 约束首尾边界条件
        // std::cout << "hard: " << end_constrain_ << std::endl;
    }
    else if (end_constrain_ == SOFT_CONSTRAINT)
    {
        variable_num_ = 3 * (control_points_.rows() - start_id_); // 需优化的控制点数量, 只约束起始边界条件
        // std::cout << "soft: " << end_constrain_ << std::endl;
    }

    min_cost_ = std::numeric_limits<double>::max();

    nlopt::opt opt(nlopt::algorithm(algorithm_), variable_num_); // algorithm_ = 11 -> LBFGS

    opt.set_min_objective(BsplineOptimizer::costFunction, this); // this指针为BsplineOptimizer::costFunction提供额外数据
    opt.set_maxeval(max_iteration_num_); // 100
    // opt.set_xtol_rel(1e-4);
    // opt.set_maxtime(1e-2);

    /* ---------- init variables ---------- */
    vector<double> q(variable_num_); // 待优化变量x
    double final_cost;
    for (int i = 0; i < int(control_points_.rows()); ++i)
    {
        if (i < start_id_)
            continue;

        if (end_constrain_ == HARD_CONSTRAINT && i >= end_id_)
        {
            continue;
            // std::cout << "jump" << std::endl;
        }

        for (int j = 0; j < 3; j++) // x, y, z轴
            q[3 * (i - start_id_) + j] = control_points_(i, j); // 将优化变量转换为Nlopt要求的格式
    }

    if (end_constrain_ == SOFT_CONSTRAINT) // end_pt_为最后一段B-spline的起点(t = 0), 用于惩罚软约束下轨迹末端的位置
    {
        end_pt_ = (1 / 6.0) *
                  (control_points_.row(control_points_.rows() - 3) + 4 * control_points_.row(control_points_.rows() - 2) +
                   control_points_.row(control_points_.rows() - 1));
        // std::cout << "end pt" << std::endl;
    }

    try
    {
        /* ---------- optimization ---------- */
        cout << "[Optimization]: begin-------------" << endl;
        cout << fixed << setprecision(7); // 设置精度
        vec_time_.clear();
        vec_cost_.clear();
        time_start_ = ros::Time::now();

        nlopt::result result = opt.optimize(q, final_cost);

        /* ---------- get results ---------- */
        std::cout << "[Optimization]: iter num: " << iter_num_ << std::endl;
        // cout << "Min cost:" << min_cost_ << endl;

        // 将Nlopt计算的最优值转换为Eigen::MatrixXd格式
        for (int i = 0; i < control_points_.rows(); ++i)
        {
            if (i < start_id_)
                continue;

            if (end_constrain_ == HARD_CONSTRAINT && i >= end_id_)
                continue;

            for (int j = 0; j < 3; j++)
                control_points_(i, j) = best_variable_[3 * (i - start_id_) + j];
        }

        cout << "[Optimization]: end-------------" << endl;
    }
    catch (std::exception &e)
    {
        cout << "[Optimization]: nlopt exception: " << e.what() << endl;
    }
}

void BsplineOptimizer::calcSmoothnessCost(const vector<Eigen::Vector3d> &q, double &cost,
                                          vector<Eigen::Vector3d> &gradient)
{
    cost = 0.0;
    std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

    Eigen::Vector3d jerk; // Σ|Qi+3 - 3 * Qi+2 + 3 * Qi+1 - Qi|^2

    for (int i = 0; i < q.size() - order_; i++) // 0 <= i <= N-3
    {
        /* evaluate jerk */
        jerk = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]; // 这里是将每个控制点的x, y, z轴联合起来评估
        cost += jerk.squaredNorm();

        /* jerk gradient */
        gradient[i + 0] += 2.0 * jerk * (-1.0);
        gradient[i + 1] += 2.0 * jerk * (3.0);
        gradient[i + 2] += 2.0 * jerk * (-3.0);
        gradient[i + 3] += 2.0 * jerk * (1.0);
    }
}

void BsplineOptimizer::calcDistanceCost(const vector<Eigen::Vector3d> &q, double &cost,
                                        vector<Eigen::Vector3d> &gradient)
{
    cost = 0.0;
    std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

    double dist;
    Eigen::Vector3d dist_grad, g_zero(0, 0, 0);

    int end_idx = end_constrain_ == SOFT_CONSTRAINT ? q.size() : q.size() - order_;

    // 前p个控制点始终不优化, 后p个控制点是否优化取决于SOFT_CONSTRAINT or HARD_CONSTRAINT: SOFT->优化; HARD->不优化
    for (int i = order_; i < end_idx; i++)
    {
        if (!dynamic_)
        {
            edt_env_->evaluateEDTWithGrad(q[i], -1.0, dist, dist_grad); // 得到某个控制点的距离障碍的最小距离和梯度值
        }
        else
        {
            double time = double(i + 2 - order_) * bspline_interval_ + time_traj_start_;
            edt_env_->evaluateEDTWithGrad(q[i], time, dist, dist_grad);
        }

        cost += dist < dist0_ ? pow(dist - dist0_, 2) : 0.0; // dist0_ = 0.8, 为惩罚距离的阈值
        gradient[i] += dist < dist0_ ? 2.0 * (dist - dist0_) * dist_grad : g_zero; // 这里没有交叉项, += 写成 = 也可以
    }
}

void BsplineOptimizer::calcFeasibilityCost(const vector<Eigen::Vector3d> &q, double &cost,
                                           vector<Eigen::Vector3d> &gradient)
{
    cost = 0.0;
    std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

    /* ---------- abbreviation ---------- */
    double ts, vm2, am2, ts_inv2, ts_inv4;
    vm2 = max_vel_ * max_vel_;
    am2 = max_acc_ * max_acc_;

    ts = bspline_interval_;
    ts_inv2 = 1 / ts / ts;
    ts_inv4 = ts_inv2 * ts_inv2;

    /* ---------- velocity feasibility ---------- */
    for (int i = 0; i < q.size() - 1; i++)
    {
        Eigen::Vector3d vi = q[i + 1] - q[i]; // Vi = 1 / ts * (Qi+1 - Qi)
        for (int j = 0; j < 3; j++) // x, y, z轴
        {
            double vd = vi(j) * vi(j) * ts_inv2 - vm2; // B-spline把所有速度控制点限制在max_vel_以下, 就可以保证速度的可行性
            cost += vd > 0.0 ? pow(vd, 2) : 0.0;

            gradient[i + 0](j) += vd > 0.0 ? 2.0 * vd * ts_inv2 * (-2.0) * vi(j) : 0.0; // 求梯度要对Qi
            gradient[i + 1](j) += vd > 0.0 ? 2.0 * vd * ts_inv2 * (2.0) * vi(j) : 0.0;
        }
    }

    /* ---------- acceleration feasibility ---------- */
    for (int i = 0; i < q.size() - 2; i++)
    {
        Eigen::Vector3d ai = q[i + 2] - 2 * q[i + 1] + q[i]; // Ai = 1 / ts * (Vi+1 - Vi) = 1 / ts^2 * (Qi+2 - 2* Qi+1 + Qi)
        for (int j = 0; j < 3; j++)
        {
            double ad = ai(j) * ai(j) * ts_inv4 - am2;
            cost += ad > 0.0 ? pow(ad, 2) : 0.0;

            gradient[i + 0](j) += ad > 0.0 ? 2.0 * ad * ts_inv4 * (2.0) * ai(j) : 0.0;
            gradient[i + 1](j) += ad > 0.0 ? 2.0 * ad * ts_inv4 * (-4.0) * ai(j) : 0.0;
            gradient[i + 2](j) += ad > 0.0 ? 2.0 * ad * ts_inv4 * (2.0) * ai(j) : 0.0;
        }
    }
}

void BsplineOptimizer::calcEndpointCost(const vector<Eigen::Vector3d> &q, double &cost,
                                        vector<Eigen::Vector3d> &gradient)
{
    cost = 0.0;
    std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

    // zero cost and gradient in hard constraints
    if (end_constrain_ == SOFT_CONSTRAINT)
    {
        Eigen::Vector3d q_3, q_2, q_1, qd;
        q_3 = q[q.size() - 3];
        q_2 = q[q.size() - 2];
        q_1 = q[q.size() - 1];

        // pos = 1 / 6 * (Qi + 4 * Qi+1 + Qi+2)
        qd = 1 / 6.0 * (q_3 + 4 * q_2 + q_1) - end_pt_; // 惩罚最后一段轨迹的起点和endPoint之间的距离
        cost += qd.squaredNorm();

        gradient[q.size() - 3] += 2 * qd * (1 / 6.0);
        gradient[q.size() - 2] += 2 * qd * (4 / 6.0);
        gradient[q.size() - 1] += 2 * qd * (1 / 6.0);
    }
}

// 评估给定x点处的梯度grad和对应的目标函数的值f_combine, 提供给Nlopt进行下一次迭代
void BsplineOptimizer::combineCost(const std::vector<double> &x, std::vector<double> &grad, double &f_combine)
{
    /* ---------- convert to control point vector ---------- */
    vector<Eigen::Vector3d> q; // q为优化后的控制点(优化变量+固定变量), 用来评估此次迭代后的cost
    // q.resize(control_points_.rows()); // 错误写法, resize后应使用[index]访问, 不能push_back, 先resize然后用索引访问效率最高
    q.reserve(control_points_.rows()); // 先用reserve分配内存, 再push_back比直接push_back效率高 1/2 ~ 2/3 倍左右

    /* first p points */
    for (int i = 0; i < order_; i++)
        q.push_back(control_points_.row(i));

    /* optimized control points */
    for (int i = 0; i < variable_num_ / 3; i++)
    {
        Eigen::Vector3d qi(x[3 * i], x[3 * i + 1], x[3 * i + 2]); // 优化变量x为std::vector<double>, 需要转化为Vector3d格式
        q.push_back(qi);
    }

    /* last p points */
    if (end_constrain_ == END_CONSTRAINT::HARD_CONSTRAINT) // 若为SOFT_CONSTRAINT, 控制点就已经包含在上面variable_num_中了
    {
        for (int i = 0; i < order_; i++)
            q.push_back(control_points_.row(control_points_.rows() - order_ + i)); // 注意row下标是从0开始
    }

    /* ---------- evaluate cost and gradient ---------- */
    double f_smoothness, f_distance, f_feasibility, f_endpoint;

    vector<Eigen::Vector3d> g_smoothness, g_distance, g_feasibility, g_endpoint; // 每个控制点的梯度向量
    g_smoothness.resize(control_points_.rows());
    g_distance.resize(control_points_.rows());
    g_feasibility.resize(control_points_.rows());
    g_endpoint.resize(control_points_.rows());

    calcSmoothnessCost(q, f_smoothness, g_smoothness); // 计算fs的cost和gradient
    calcDistanceCost(q, f_distance, g_distance); // 计算fc的cost和gradient
    calcFeasibilityCost(q, f_feasibility, g_feasibility); // 计算fv, fa的cost和gradient
    calcEndpointCost(q, f_endpoint, g_endpoint); // 计算Endpoint的cost和gradient(在软约束时将轨迹终点向目标点引导)

    /* ---------- convert to NLopt format ---------- */
    grad.resize(variable_num_); // NLopt中格式为std::vector<double>

    f_combine = lamda1_ * f_smoothness + lamda2_ * f_distance + lamda3_ * f_feasibility + lamda4_ * f_endpoint;

    for (int i = 0; i < variable_num_ / 3; i++)
        for (int j = 0; j < 3; j++)
        {
            /* the first p points is static here */
            // 只包含需要优化的变量的梯度(size = variable_num_)
            grad[3 * i + j] = lamda1_ * g_smoothness[i + order_](j) + lamda2_ * g_distance[i + order_](j) +
                              lamda3_ * g_feasibility[i + order_](j) + lamda4_ * g_endpoint[i + order_](j);
        }

    /* ---------- print cost ---------- */
    iter_num_ += 1;

    if (iter_num_ % 100 == 0)
    {
        // cout << iter_num_ << " smooth: " << lamda1_ * f_smoothness << " , dist: " << lamda2_ * f_distance
        //      << ", fea: " << lamda3_ * f_feasibility << ", end: " << lamda4_ * f_endpoint << ", total: " << f_combine
        //      << endl;
    }
}

double BsplineOptimizer::costFunction(const std::vector<double> &x, std::vector<double> &grad, void *func_data)
{
    BsplineOptimizer *opt = reinterpret_cast<BsplineOptimizer *>(func_data);

    double cost;
    opt->combineCost(x, grad, cost); // 为Nlopt评估目标函数的值: J = lamda1 * fs + lamda2 * fc + lamda3 * (fv + fa), 并计算梯度

    /* save the min cost result */
    if (cost < opt->min_cost_)
    {
        opt->min_cost_ = cost; // 保存最优cost
        opt->best_variable_ = x; // 保存最优的控制点位置(std::vector<double>)
    }

    return cost;

    // /* ---------- evaluation ---------- */

    // ros::Time te1 = ros::Time::now();
    // double time_now = (te1 - opt->time_start_).toSec();
    // opt->vec_time_.push_back(time_now);
    // if (opt->vec_cost_.size() == 0)
    // {
    //   opt->vec_cost_.push_back(f_combine);
    // }
    // else if (opt->vec_cost_.back() > f_combine)
    // {
    //   opt->vec_cost_.push_back(f_combine);
    // }
    // else
    // {
    //   opt->vec_cost_.push_back(opt->vec_cost_.back());
    // }
}

} // namespace dyn_planner