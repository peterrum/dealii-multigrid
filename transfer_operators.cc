// Test different partitioning strategies and their influence on the performance
// of the transfer operators.

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

namespace dealii::MGTools
{
  template <typename TransferType>
  void
  run_transfer_test(
    const TransferType transfer,
    const MGLevelObject<std::shared_ptr<const Utilities::MPI::Partitioner>>
      &partitioners)
  {
    using Number     = double;
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    std::vector<double> ts_prolongation;
    std::vector<double> ts_restriction;

    const auto comm =
      partitioners[partitioners.min_level()]->get_mpi_communicator();

    for (unsigned int l = partitioners.min_level();
         l < partitioners.max_level();
         ++l)
      {
        VectorType vector_coarse(partitioners[l]);
        VectorType vector_fine(partitioners[l + 1]);

        const auto timed_execution = [&](const auto &fu) {
          {
            for (unsigned int i = 0; i < 10; ++i)
              fu();
          }

          double time = 0.0;

          MPI_Barrier(comm);

          {
            ScopedTimer timer(time);
            for (unsigned int i = 0; i < 100; ++i)
              fu();
          }

          return time;
        };

        const auto t_prolongation = timed_execution([&]() {
          transfer.prolongate_and_add(l + 1, vector_fine, vector_coarse);
        });
        const auto t_restriction  = timed_execution([&]() {
          transfer.restrict_and_add(l + 1, vector_coarse, vector_fine);
        });

        ts_prolongation.emplace_back(t_prolongation);
        ts_restriction.emplace_back(t_restriction);
      }

    std::vector<double> ts_prolongation_min(ts_prolongation.size());
    std::vector<double> ts_prolongation_max(ts_prolongation.size());
    std::vector<double> ts_prolongation_sum(ts_prolongation.size());
    std::vector<double> ts_restriction_min(ts_restriction.size());
    std::vector<double> ts_restriction_max(ts_restriction.size());
    std::vector<double> ts_restriction_sum(ts_restriction.size());

    Utilities::MPI::min(ts_prolongation, comm, ts_prolongation_min);
    Utilities::MPI::max(ts_prolongation, comm, ts_prolongation_max);
    Utilities::MPI::sum(ts_prolongation, comm, ts_prolongation_sum);
    Utilities::MPI::min(ts_restriction, comm, ts_restriction_min);
    Utilities::MPI::max(ts_restriction, comm, ts_restriction_max);
    Utilities::MPI::sum(ts_restriction, comm, ts_restriction_sum);

    const unsigned int n_proc = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int rank   = Utilities::MPI::this_mpi_process(comm);

    if (rank == 0)
      {
        ConvergenceTable table;

        for (unsigned int i = 0; i < ts_prolongation.size(); ++i)
          {
            table.add_value("level", i);
            table.add_value("prol_min", ts_prolongation_min[i]);
            table.set_scientific("prol_min", true);
            table.add_value("prol_max", ts_prolongation_max[i]);
            table.set_scientific("prol_max", true);
            table.add_value("prol_avg", ts_prolongation_sum[i] / n_proc);
            table.set_scientific("prol_avg", true);
            table.add_value("rest_min", ts_restriction_min[i]);
            table.set_scientific("rest_min", true);
            table.add_value("rest_max", ts_restriction_max[i]);
            table.set_scientific("rest_max", true);
            table.add_value("rest_avg", ts_restriction_sum[i] / n_proc);
            table.set_scientific("rest_avg", true);
          }

        table.write_text(std::cout, ConvergenceTable::org_mode_table);

        std::cout << std::endl;
      }
  }

  template <int dim, int spacedim>
  void
  run_transfer_test(const Triangulation<dim, spacedim> &tria,
                    const unsigned int                  fe_degree_fine,
                    const unsigned int                  n_components)
  {
    using Number     = double;
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    const FESystem<dim> fe(FE_Q<dim>{fe_degree_fine}, n_components);

    DoFHandler<dim, spacedim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    const unsigned int min_level = 0;
    const unsigned int max_level = tria.n_global_levels() - 1;

    MGConstrainedDoFs mg_constrained_dofs;

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                       dirichlet_boundary);

    MGLevelObject<AffineConstraints<Number>> constraints(min_level, max_level);

    MGLevelObject<std::shared_ptr<const Utilities::MPI::Partitioner>>
      partitioners(min_level, max_level);

    for (auto l = min_level; l <= max_level; ++l)
      {
        auto &constraint = constraints[l];

        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                      l,
                                                      relevant_dofs);
        constraint.reinit(relevant_dofs);
        constraint.add_lines(mg_constrained_dofs.get_boundary_indices(l));
        constraint.close();


    typename MatrixFree<dim, Number>::AdditionalData data;
    data.mg_level             = l;

        MatrixFree<dim, Number> matrix_free;
        matrix_free.reinit(dof_handler, constraint, QGauss<dim>(fe_degree_fine), data);

        partitioners[l] = matrix_free.get_vector_partitioner();
      }

    MGTransferMatrixFree<dim, Number> transfer(mg_constrained_dofs);
    {
      std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
        external_partitioners;

      for (auto l = min_level; l <= max_level; ++l)
        external_partitioners.emplace_back(partitioners[l]);
        
    transfer.build(dof_handler, external_partitioners);
    }

    run_transfer_test(transfer, partitioners); // run tests
  }

  template <int dim, int spacedim>
  void
  run_transfer_test(
    const std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
      &                trias,
    const unsigned int fe_degree_fine,
    const unsigned int n_components)
  {
    using Number     = double;
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    std::string type = "HMG";

    const auto level_degrees =
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        fe_degree_fine,
        MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
          bisect);

    const unsigned int min_level = 0;
    const unsigned int max_level =
      (type == "PMG" ? level_degrees.size() : trias.size()) - 1;
    ;

    MGLevelObject<DoFHandler<dim>>           dof_handlers(min_level, max_level);
    MGLevelObject<AffineConstraints<double>> constraints(min_level, max_level);
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers(min_level,
                                                                 max_level);

    MappingQ1<dim> mapping;

    MGLevelObject<std::shared_ptr<const Utilities::MPI::Partitioner>>
      partitioners(min_level, max_level);

    for (auto l = min_level; l <= max_level; ++l)
      {
        auto &dof_handler = dof_handlers[l];
        auto &constraint  = constraints[l];

        dof_handler.reinit(type == "PMG" ? *trias.back() : *trias[l]);
        dof_handler.distribute_dofs(FESystem<dim>(
          FE_Q<dim>{type == "PMG" ? level_degrees[l] : level_degrees.back()},
          n_components));

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                locally_relevant_dofs);
        constraint.reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(mapping,
                                                 dof_handler,
                                                 0,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 constraint);
        DoFTools::make_hanging_node_constraints(dof_handler, constraint);
        constraint.close();

        MatrixFree<dim, double> matrix_free;
        matrix_free.reinit(dof_handler, constraint, QGauss<dim>(dof_handler.get_fe().degree + 1));

        partitioners[l] = matrix_free.get_vector_partitioner();
      }

    for (unsigned int l = min_level; l < max_level; ++l)
      transfers[l + 1].reinit(dof_handlers[l + 1],
                              dof_handlers[l],
                              constraints[l + 1],
                              constraints[l]);

    MGTransferGlobalCoarsening<dim, VectorType> transfer(transfers, [&](const auto l, auto &vec) {
      vec.reinit(partitioners[l]);;
    });

    run_transfer_test(transfer, partitioners); // run tests
  }
} // namespace dealii::MGTools

void
run(const unsigned int n_ref_global, const unsigned int fe_degree_fine)
{
  const unsigned int dim          = 3;
  const unsigned int spacedim     = dim;
  const unsigned int n_components = 1;

  parallel::distributed::Triangulation<dim, spacedim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim, spacedim>::MeshSmoothing::none,
    parallel::distributed::Triangulation<dim, spacedim>::construct_multigrid_hierarchy);
  
  GridGenerator::hyper_cube(tria, -1.0, +1.0);
  tria.refine_global(n_ref_global);

    {
  std::unique_ptr<RepartitioningPolicyTools::Base<dim>> policy =
            std::make_unique<RepartitioningPolicyTools::FirstChildPolicy<dim>>(
              tria);

      const auto trias =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          tria, *policy, true, true);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "GLOBAL COARSENING:" << std::endl;

      // MGTools::print_multigrid_statistics(trias);
      MGTools::run_transfer_test(trias, fe_degree_fine, n_components);
    }

    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "LOCAL SMOOTHING:" << std::endl;

      // MGTools::print_multigrid_statistics(tria);
      MGTools::run_transfer_test(tria, fe_degree_fine, n_components);
    }

}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int n_ref_global = argc > 1 ? atoi(argv[1]) : 8;
  const unsigned int degree       = argc > 2 ? atoi(argv[2]) : 4;

  run(n_ref_global, degree);
}