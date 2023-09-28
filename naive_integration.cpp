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
constexpr double SIMTIME = 100000;
constexpr double BASE_DT = 0.1;
// used to initialize spike arrays
constexpr std::size_t BASE_SIM_STEPS = static_cast<std::size_t>( SIMTIME / BASE_DT );

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


struct OneStateNeuron
{

  //! neuron state, must be C-array for GSL solver
  double y[ 1 ] = {0};

  double g = 0;
  double I_syn = 0;

  Parameters P;
};


struct TwoStateNeuron
{
  enum StateVecElems
  {
    V_m = 0,
    g,
    STATE_VEC_SIZE
  };

  //! neuron state, must be C-array for GSL solver
  double y[ STATE_VEC_SIZE ] = {0, 0};
  double I_syn = 0;

  Parameters P;
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
  const double I_syn_exc = - y[ TwoStateNeuron::g ] * ( y[0] - node.P.E_ex );
  const double I_L = node.P.g_L * ( y[0] - node.P.E_L );

  node.I_syn = I_syn_exc;

  f[ 0 ] = ( -I_L + I_syn_exc ) / node.P.C_m;
  f[ 1 ] = -y[ TwoStateNeuron::g ] / node.P.tau_syn;

 
  return GSL_SUCCESS;
}


SimulationResults oneStateSimulation( double dt, 
                                      std::array<double, BASE_SIM_STEPS> &spikes,
                                      std::array<long, BASE_SIM_STEPS> spike_indices,
                                      std::array<double, BASE_SIM_STEPS> &spike_times )
{
  long sim_steps = static_cast<long>( SIMTIME / dt );
  std::cout << "Num steps: " << sim_steps << std::endl;

  // Determine which time steps of current simulation to input spikes
  long time_scale = std::lround( BASE_DT / dt);
  for ( auto &item : spike_indices ) {
    item *= time_scale;
  }
  std::cout << "Time scale: " << time_scale << std::endl;

  OneStateNeuron nrn;
  double f [ 2 ] = {0, 0}; // array containing derivatives of y

  // Set up GSL variables
  gsl_odeiv_system sys; //!< struct describing system
  sys.function = oneStateDynamics;
  sys.jacobian = nullptr;
  sys.dimension = 1;
  sys.params = &nrn;

  gsl_odeiv_step* s_ = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, 1 );
  gsl_odeiv_control* c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  gsl_odeiv_evolve* e_ = gsl_odeiv_evolve_alloc( 1 );

  double IntegrationStep_ = dt;

  SimulationResults results( sim_steps + 1 );

  long spike_index = 0; // variable to access spikes array
  long step_idx = 0;
  double t = 0.0;
  // Main simulation loop
  while ( step_idx < sim_steps )
  {
    // incoming spikes
    if ( step_idx == spike_indices[ spike_index ] ) {
      nrn.g += spikes[ spike_index ];
      ++spike_index;
    }

    results.V_m.push_back( nrn.y[ 0 ] );     
    results.g.push_back( nrn.g );     
    results.I_syn.push_back( nrn.I_syn  );     
    results.times.push_back( step_idx * dt );     

    double t_ = 0; 
    //
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
        nrn.y
        );              // neuronal state
      if ( status != GSL_SUCCESS )
        goto outOfLoop; 
    }  
    nrn.g *= std::exp( -dt / nrn.P.tau_syn );

    ++step_idx;
    t += dt;
  }

  // last step
  results.V_m.push_back( nrn.y[ 0 ] );     
  results.I_syn.push_back( nrn.I_syn );     
  results.g.push_back( nrn.g );     
  results.times.push_back( step_idx * dt );     
  return results;

  outOfLoop:
    std::cout << "GSL failed" << std::endl;
    return results;
}


SimulationResults twoStateSimulation( double dt, 
                                      std::array<double, BASE_SIM_STEPS> &spikes,
                                      std::array<long, BASE_SIM_STEPS> spike_indices,
                                      std::array<double, BASE_SIM_STEPS> &spike_times )
{
  long sim_steps = static_cast<long>( SIMTIME / dt );
  std::cout << "Num steps: " << sim_steps << std::endl;

  // Determine which time steps of current simulation to input spikes
  long time_scale = std::lround( BASE_DT / dt);
  for ( auto &item : spike_indices ) {
    item *= time_scale;
  }
  std::cout << "Time scale: " << time_scale << std::endl;

  TwoStateNeuron nrn;
  double f [ 2 ] = {0, 0}; // array containing derivatives of y

  // Set up GSL variables
  gsl_odeiv_system sys; //!< struct describing system
  sys.function = twoStateDynamics;
  sys.jacobian = nullptr;
  sys.dimension = TwoStateNeuron::STATE_VEC_SIZE;
  sys.params = &nrn;

  gsl_odeiv_step* s_ = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, TwoStateNeuron::STATE_VEC_SIZE );
  gsl_odeiv_control* c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  gsl_odeiv_evolve* e_ = gsl_odeiv_evolve_alloc( TwoStateNeuron::STATE_VEC_SIZE );

  double IntegrationStep_ = dt;

  SimulationResults results( sim_steps + 1 );

  long spike_index = 0; // variable to access spikes array
  long step_idx = 0;
  double t = 0.0;
  // Main simulation loop
  while ( step_idx < sim_steps )
  {
    // incoming spikes
    if ( step_idx == spike_indices[ spike_index ] ) {
      nrn.y[ 1 ] += spikes[ spike_index ];
      ++spike_index;
    }

    results.V_m.push_back( nrn.y[ 0 ] );     
    results.g.push_back( nrn.y[ 1 ] );     
    results.I_syn.push_back( nrn.I_syn );     
    results.times.push_back( step_idx * dt );     

    double t_ = 0; 
    //
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
        nrn.y
        );              // neuronal state
      if ( status != GSL_SUCCESS )
        goto outOfLoop; 
    }  

    ++step_idx;
    t += dt;
  }

  // last step
  results.V_m.push_back( nrn.y[ 0 ] );     
  results.g.push_back( nrn.y[ 1 ] );     
  results.I_syn.push_back( nrn.I_syn );     
  results.times.push_back( step_idx * dt );     


  return results;

  outOfLoop:
    std::cout << "GSL failed" << std::endl;
    return results;
}


int main() {
  // set up input to neuron
  double w = 0.001;
  double rate = BASE_DT * 1000;
  std::default_random_engine generator;
  generator.seed(1234);
  std::poisson_distribution<int> distribution( rate );
  
  std::array<double, BASE_SIM_STEPS> spikes;
  for ( auto& item: spikes )
  {
    item = distribution(generator) * w;    
  }

  std::array<double, BASE_SIM_STEPS> spike_times;
  for ( long i = 0; i < spike_times.size(); i++ )
  {
    spike_times[ i ] = i * BASE_DT;     
  }

  // reference indices for simulation with dt=BASE_DT.
  // this array will be used for determining the correct
  // time to insert spikes by dividing by dt.
  // this avoids comparing floats
  std::array<long, BASE_SIM_STEPS> spike_indices;
  for ( long i = 0; i < spike_times.size(); i++ )
  {
    spike_indices[ i ] = i;     
  }


  SimulationResults results1;
  SimulationResults results2;
  auto t1 = std::chrono::high_resolution_clock::now();
  results1 = twoStateSimulation( 0.1, spikes, spike_indices, spike_times );
  auto t2 = std::chrono::high_resolution_clock::now();
  results2 = oneStateSimulation( 0.1, spikes, spike_indices, spike_times );
  auto t3 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_two_state = t2 - t1;
  std::chrono::duration<double, std::milli> time_one_state = t3 - t2;

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

//   auto p1 = matplot::plot( results1.times, results1.V_m, results2.times, results2.V_m );
//   p1[0]->line_width(2);
//   p1[1]->line_width(2);
//   p1[0]->marker(matplot::line_spec::marker_style::asterisk);
//   p1[1]->marker(matplot::line_spec::marker_style::asterisk);
//   matplot::show();

  return 0;
}

