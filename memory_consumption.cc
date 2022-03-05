// Test different partitioning strategies and their influence on the performance
// of the transfer operators.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/cell_id_translator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/cell_weights.h"
#include "include/grid_generator.h"
#include "include/mg_tools.h"
#include "include/scoped_timer.h"

using namespace dealii;


void
monitor(const std::string &label)
{
  dealii::Utilities::System::MemoryStats stats;
  dealii::Utilities::System::get_memory_stats(stats);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "MONITOR " << label << ": ";

  const auto print = [&pcout](const double value) {
    const auto min_max_avg =
      dealii::Utilities::MPI::min_max_avg(value / 1e6, MPI_COMM_WORLD);

    pcout << min_max_avg.min << " " << min_max_avg.max << " " << min_max_avg.avg
          << " " << min_max_avg.sum << " ";
  };

  print(stats.VmPeak);
  print(stats.VmSize);
  print(stats.VmHWM);
  print(stats.VmRSS);

  pcout << std::endl;
}

void
run(const unsigned int fe_degree)
{
  const unsigned int dim          = 3;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  for(unsigned int n_refinements = 4; n_refinements < 20; ++n_refinements)
  {

  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim>::MeshSmoothing::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

    monitor("empty");
  
  GridGenerator::create_quadrant(tria, n_refinements);

    monitor("tria");

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

    monitor("dofhandler");

    dof_handler.distribute_mg_dofs();

    monitor("dofhandler_mg");

    pcout << "# " << n_refinements << " " << tria.n_global_active_cells() << " " << dof_handler.n_dofs() << std::endl;
  }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int degree       = argc > 1 ? atoi(argv[1]) : 4;

  run(degree);
}