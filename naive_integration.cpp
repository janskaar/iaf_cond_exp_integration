#include <iostream>
#include <random>
#include <array>
#include <cmath>
#include <matplot/matplot.h>
#include <chrono>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// Time variables
constexpr double SIMTIME = 1000000;
constexpr double BASE_DT = 0.1;
// used to initialize spike arrays
constexpr std::size_t BASE_SIM_STEPS = static_cast<std::size_t>( SIMTIME / BASE_DT );
constexpr double RATE = BASE_DT * 1000;
constexpr double WEIGHT = 0.001;

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
   , tau_syn( 2.0 )    // ms
{
}
};

struct Neuron
{
  double g = 0;
  double I_syn = 0;
  double V_m = 0;

  Parameters P;
};


struct OneStateNeuron : Neuron
{
  //! neuron state, must be C-array for GSL solver
  double f [ 1 ] = {0};
  double g_prop;
  OneStateNeuron( double g_prop_ ) {
    g_prop = g_prop_;
  }
};


struct TwoStateNeuron : Neuron
{
  //! neuron state, must be C-array for GSL solver
  double f [ 2 ] = {0, 0}; // array containing derivatives of y
};


struct SimulationResults
{
  std::vector<double> V_m;
  std::vector<double> I_syn;
  std::vector<double> g;
  std::vector<double> times;

  SimulationResults( long num ) {
    V_m.reserve(num);
    I_syn.reserve(num);
    g.reserve(num);
    times.reserve(num);
  }

  SimulationResults( ) {
  }
};


extern "C" inline int oneStateDynamics_gprop( double t, const double y[], double f[], void* pnode )
{

  OneStateNeuron& node = *( reinterpret_cast< OneStateNeuron* >( pnode ) );

  double g = node.g * 0.975;

  // compute derivatives
  const double I_syn_exc = -g * ( y[0] - node.P.E_ex );
  const double I_L = node.P.g_L * ( y[0] - node.P.E_L );

  node.I_syn = I_syn_exc;

  f[ 0 ] = ( -I_L + I_syn_exc ) / node.P.C_m;
  return GSL_SUCCESS;
}




extern "C" inline int oneStateDynamics( double t, const double y[], double f[], void* pnode )
{

  OneStateNeuron& node = *( reinterpret_cast< OneStateNeuron* >( pnode ) );

  double g = node.g * std::exp( -t / node.P.tau_syn );

  // compute derivatives
  const double I_syn_exc = -g * ( y[0] - node.P.E_ex );
  const double I_L = node.P.g_L * ( y[0] - node.P.E_L );

  node.I_syn = I_syn_exc;

  f[ 0 ] = ( -I_L + I_syn_exc ) / node.P.C_m;
  return GSL_SUCCESS;
}


extern "C" inline int twoStateDynamics( double t, const double y[], double f[], void* pnode )
{

  TwoStateNeuron& node = *( reinterpret_cast< TwoStateNeuron* >( pnode ) );

  // compute derivatives
  const double I_syn_exc = - y[ 1 ] * ( y[0] - node.P.E_ex );
  const double I_L = node.P.g_L * ( y[0] - node.P.E_L );

  node.I_syn = I_syn_exc;

  f[ 0 ] = ( -I_L + I_syn_exc ) / node.P.C_m;
  f[ 1 ] = -y[ 1 ] / node.P.tau_syn;
 
  return GSL_SUCCESS;
}


SimulationResults simulation( double dt,
                              gsl_odeiv_system sys,
                              Neuron& nrn,
                              int dim,
                              bool save_data )
{
  double y[ dim ];
  long sim_steps = static_cast<long>( SIMTIME / dt );
  std::cout << "Num steps: " << sim_steps << std::endl;

  std::default_random_engine generator;
  generator.seed(1234);
  std::poisson_distribution<int> distribution( RATE );
  
  gsl_odeiv_step* s_ = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, dim );
  gsl_odeiv_control* c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  gsl_odeiv_evolve* e_ = gsl_odeiv_evolve_alloc( dim );

  double IntegrationStep_ = dt;

  SimulationResults results( sim_steps + 1 );

  long step_idx = 0;
  double t = 0.0;
  // Main simulation loop
  while ( step_idx < sim_steps )
  {
    // incoming spikes
    if ( dim == 1 ) {
      nrn.g += distribution(generator) * WEIGHT;
    }
    else {
      y[ 1 ] += distribution(generator) * WEIGHT;
    }

    if ( save_data ) {
      results.V_m.push_back( nrn.V_m );     
      results.I_syn.push_back( nrn.I_syn  );     
      results.times.push_back( step_idx * dt );     
      if ( dim == 1 ) {
        results.g.push_back( nrn.g );     
      }
      else {
        results.g.push_back( y[ 1 ] );     
      }
    }


    double t_ = 0; 
    // Advance time step dt
    while ( t_ < dt )
    {
      const int status = gsl_odeiv_evolve_apply(
        e_,
        c_, s_,
        &sys,             // system of ODE
        &t_,                   // from t
        dt,             // to t <= step
        &IntegrationStep_, // integration step size
        y
        );              // neuronal state
      if ( status != GSL_SUCCESS )
        goto outOfLoop; 
    }  

    if ( dim == 1 ) {
      nrn.g *= 0.975;//std::exp( -dt / nrn.P.tau_syn );
    }
    else {
      nrn.g = y[ 1 ];
    }

    nrn.V_m = y[ 0 ];
    ++step_idx;
    t += dt;
  }

  // last step
  if ( save_data ) {
    results.V_m.push_back( nrn.V_m );     
    results.I_syn.push_back( nrn.I_syn  );     
    results.times.push_back( step_idx * dt );     
    if ( dim == 1 ) {
      results.g.push_back( nrn.g );     
    }
    else {
      results.g.push_back( y[ 1 ] );     
    }
  }

  return results;

  outOfLoop:
    std::cout << "GSL failed" << std::endl;
    return results;
}


int main() {
  bool save_data = false;
  Parameters p;

  double g_prop_ = std::exp( - BASE_DT / p.tau_syn );
  OneStateNeuron nrn1( g_prop_ );

  // Set up GSL variables
  gsl_odeiv_system sys1; //!< struct describing system
  sys1.function = oneStateDynamics_gprop;
  sys1.jacobian = nullptr;
  sys1.dimension = 1;
  sys1.params = &nrn1;


  SimulationResults results1;
  auto t1 = std::chrono::high_resolution_clock::now();
  results1 = simulation( 0.1, sys1, nrn1, 1 , save_data);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> time_one_state = t2 - t1;

  TwoStateNeuron nrn2;
  // Set up GSL variables
  gsl_odeiv_system sys2; //!< struct describing system
  sys2.function = twoStateDynamics;
  sys2.jacobian = nullptr;
  sys2.dimension = 2;
  sys2.params = &nrn2;


  SimulationResults results2;
  t1 = std::chrono::high_resolution_clock::now();
  results2 = simulation( 0.1, sys2, nrn2, 2, save_data );
  t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> time_two_state = t2 - t1;


  std::cout << "====================" << std::endl;
  std::cout << "Time spent one-state: " << time_one_state.count() << std::endl;
  std::cout << "Time spent two-state: " << time_two_state.count() << std::endl;
  std::cout << "====================" << std::endl;
  
//   for ( long i = 0 ; i < results1.V_m.size() ; i++ ) {
//     std::cout << "===================="  << std::endl;
//     std::cout << "V_m: " << results2.V_m[ i ]  << std::endl;
//     std::cout << "I_syn: " << results2.I_syn[ i ]  << std::endl;
//     std::cout << "===================="  << std::endl;
//   }    

  if ( save_data ) {
    auto p1 = matplot::plot( results1.times, results1.g, results1.times, results2.g);
    p1[0]->line_width(2);
    p1[1]->line_width(2);
    p1[0]->marker(matplot::line_spec::marker_style::asterisk);
    p1[1]->marker(matplot::line_spec::marker_style::asterisk);
    matplot::show();
  }

  return 0;
}

