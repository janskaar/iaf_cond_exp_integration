#include <iostream>
#include <random>
#include <array>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

struct Parameters
{
  double V_th_;    //!< Threshold Potential in mV
  double V_reset;  //!< Reset Potential in mV
  double t_ref;    //!< Refractory period in ms
  double g_L;      //!< Leak Conductance in nS
  double C_m;      //!< Membrane Capacitance in pF
  double E_ex;     //!< Excitatory reversal Potential in mV
  double E_in;     //!< Inhibitory reversal Potential in mV
  double E_L;      //!< Leak reversal Potential (aka resting potential) in mV
  double tau_syn;

  Parameters()
   : V_th_( 15.0 )    // mV
   , V_reset( 0.0 )  // mV
   , t_ref( 2.0 )      // ms
   , g_L( 16.6667 )    // nS
   , C_m( 250.0 )      // pF
   , E_ex( 70.0 )       // mV
   , E_L( 0.0 )      // mV
   , tau_syn( 0.2 )    // ms
{
}
};


struct Neuron
{
  enum StateVecElems
  {
    V_m = 0,
    g,
    STATE_VEC_SIZE
  };

  //! neuron state, must be C-array for GSL solver
  double y[ STATE_VEC_SIZE ] = {0, 0};

  Parameters P;
};


extern "C" inline int dynamics( double t, const double y[], double f[], void* pnode )
{

  const Neuron& node = *( reinterpret_cast< Neuron* >( pnode ) );

  const double V = std::min( y[ Neuron::V_m ], node.P.V_th_ );

  // compute derivatives
  const double I_syn_exc = y[ Neuron::g ] * ( V - node.P.E_ex );
  const double I_L = node.P.g_L * ( V - node.P.E_L );

  f[ 0 ] = ( -I_L - I_syn_exc ) / node.P.C_m;
  f[ 1 ] = -y[ Neuron::g ] / node.P.tau_syn;

  return GSL_SUCCESS;
}


int main() {
  Neuron nrn;
  double f [ 2 ] = {0, 0}; // array containing derivatives of y

  // time variables
  constexpr double simtime = 100;
  constexpr double step_ = 0.1; // simulation step size
  constexpr long sim_steps = static_cast<long>( simtime / step_ );

  // set up input to neuron
  double w = 0.01;
  double rate = step_ * 1000;
  std::default_random_engine generator;
  generator.seed(1234);
  std::poisson_distribution<int> distribution( rate );
  std::array<int, sim_steps> spikes;
  
  for ( auto& item: spikes )
  {
    item = distribution(generator);    
  }

// Set up GSL variables
  gsl_odeiv_system sys; //!< struct describing system
  sys.function = dynamics;
  sys.jacobian = nullptr;
  sys.dimension = Neuron::STATE_VEC_SIZE;
  sys.params = &nrn;

  gsl_odeiv_step* s_ = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, Neuron::STATE_VEC_SIZE );
  gsl_odeiv_control* c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  gsl_odeiv_evolve* e_ = gsl_odeiv_evolve_alloc( Neuron::STATE_VEC_SIZE );

  double IntegrationStep_ = step_;

  
  int step_idx = 0;
// Main simulation loop
  while ( step_idx < sim_steps )
  {
    double t_ = 0; 
    while ( t_ < step_ )
    {
      const int status = gsl_odeiv_evolve_apply(
        e_,
        c_,
        s_,
        &sys,             // system of ODE
        &t_,                   // from t
        step_,             // to t <= step
        &IntegrationStep_, // integration step size
        nrn.y
        );              // neuronal state
      if ( status != GSL_SUCCESS )
        goto outOfLoop; 
    }  

    ++step_idx;
    nrn.y[ 1 ] += static_cast<double>( spikes[ step_idx ] ) * w;
    std::cout << "step idx: " << step_idx << ", V_m: " << nrn.y[ 0 ] << ", I_syn: " << nrn.y[ 1 ] << std::endl;
  }

  return 0;
  outOfLoop:
    std::cout << "GSL failed" << std::endl;
    return 0;
}

