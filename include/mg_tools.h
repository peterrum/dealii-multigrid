#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>

#include <deal.II/grid/cell_id_translator.h>
#include <deal.II/grid/grid_in.h>

namespace dealii::MGTools
{
  namespace internal
  {
    double
    workload_imbalance(
      const std::vector<types::global_dof_index> &n_cells_on_leves,
      const MPI_Comm                              comm)
    {
      std::vector<types::global_dof_index> n_cells_on_leves_max(
        n_cells_on_leves.size());
      std::vector<types::global_dof_index> n_cells_on_leves_sum(
        n_cells_on_leves.size());

      Utilities::MPI::max(n_cells_on_leves, comm, n_cells_on_leves_max);
      Utilities::MPI::sum(n_cells_on_leves, comm, n_cells_on_leves_sum);

      const unsigned int n_proc = Utilities::MPI::n_mpi_processes(comm);

      const double ideal_work = std::accumulate(n_cells_on_leves_sum.begin(),
                                                n_cells_on_leves_sum.end(),
                                                0) /
                                static_cast<double>(n_proc);
      const double workload_imbalance =
        std::accumulate(n_cells_on_leves_max.begin(),
                        n_cells_on_leves_max.end(),
                        0) /
        ideal_work;

      return workload_imbalance;
    }
  } // namespace internal

  template <int dim, int spacedim>
  std::tuple<std::vector<types::global_dof_index>, MPI_Comm>
  workload(const Triangulation<dim, spacedim> &tria)
  {
    if (const parallel::TriangulationBase<dim, spacedim> *tr =
          dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &tria))
      Assert(
        tr->is_multilevel_hierarchy_constructed(),
        ExcMessage(
          "We can only compute the workload imbalance if the multilevel hierarchy has been constructed!"));

    const unsigned int n_global_levels = tria.n_global_levels();

    std::vector<types::global_dof_index> n_cells_on_leves(n_global_levels);

    for (int lvl = n_global_levels - 1; lvl >= 0; --lvl)
      for (const auto &cell : tria.cell_iterators_on_level(lvl))
        if (cell->is_locally_owned_on_level())
          ++n_cells_on_leves[lvl];

    return {n_cells_on_leves, tria.get_communicator()};
  }

  template <int dim, int spacedim>
  std::tuple<std::vector<types::global_dof_index>, MPI_Comm>
  workload(
    const std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
      &trias)
  {
    const unsigned int n_global_levels = trias.size();

    std::vector<types::global_dof_index> n_cells_on_leves(n_global_levels);

    for (int lvl = n_global_levels - 1; lvl >= 0; --lvl)
      for (const auto &cell : trias[lvl]->active_cell_iterators())
        if (cell->is_locally_owned())
          ++n_cells_on_leves[lvl];

    return {n_cells_on_leves, trias.back()->get_communicator()};
  }

  template <int dim, int spacedim>
  std::tuple<std::vector<types::global_dof_index>,
             std::vector<types::global_dof_index>,
             MPI_Comm>
  vertical_communication_cost(const Triangulation<dim, spacedim> &tria)
  {
    const unsigned int n_global_levels = tria.n_global_levels();

    std::vector<types::global_dof_index> cells_local(n_global_levels);
    std::vector<types::global_dof_index> cells_remote(n_global_levels);

    const MPI_Comm communicator = tria.get_communicator();

    const unsigned int my_rank = Utilities::MPI::this_mpi_process(communicator);

    for (unsigned int lvl = 0; lvl < n_global_levels - 1; ++lvl)
      for (const auto &cell : tria.cell_iterators_on_level(lvl))
        if (cell->is_locally_owned_on_level() && cell->has_children())
          for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell;
               ++i)
            {
              const auto level_subdomain_id =
                cell->child(i)->level_subdomain_id();
              if (level_subdomain_id == my_rank)
                ++cells_local[lvl + 1];
              else if (level_subdomain_id != numbers::invalid_unsigned_int)
                ++cells_remote[lvl + 1];
              else
                AssertThrow(false, ExcNotImplemented());
            }

    return {cells_local, cells_remote, tria.get_communicator()};
  }

  template <int dim, int spacedim>
  std::tuple<std::vector<types::global_dof_index>,
             std::vector<types::global_dof_index>,
             MPI_Comm>
  vertical_communication_cost(
    const std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
      &trias)
  {
    const unsigned int n_global_levels = trias.size();

    std::vector<types::global_dof_index> cells_local(n_global_levels);
    std::vector<types::global_dof_index> cells_remote(n_global_levels);

    const MPI_Comm communicator = trias.back()->get_communicator();

    const unsigned int my_rank = Utilities::MPI::this_mpi_process(communicator);

    for (unsigned int lvl = 0; lvl < n_global_levels - 1; ++lvl)
      {
        const auto &tria_coarse = *trias[lvl];
        const auto &tria_fine   = *trias[lvl + 1];

        const unsigned int n_coarse_cells  = tria_fine.n_global_coarse_cells();
        const unsigned int n_global_levels = tria_fine.n_global_levels();

        const dealii::internal::CellIDTranslator<dim> cell_id_translator(
          n_coarse_cells, n_global_levels);

        IndexSet is_fine_owned(cell_id_translator.size());
        IndexSet is_fine_required(cell_id_translator.size());

        for (const auto &cell : tria_fine.active_cell_iterators())
          if (!cell->is_artificial() && cell->is_locally_owned())
            is_fine_owned.add_index(cell_id_translator.translate(cell));

        for (const auto &cell : tria_coarse.active_cell_iterators())
          if (!cell->is_artificial() && cell->is_locally_owned())
            {
              if (cell->level() + 1u == tria_fine.n_global_levels())
                continue;

              for (unsigned int i = 0;
                   i < GeometryInfo<dim>::max_children_per_cell;
                   ++i)
                is_fine_required.add_index(
                  cell_id_translator.translate(cell, i));
            }


        std::vector<unsigned int> is_fine_required_ranks(
          is_fine_required.n_elements());

        Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
          process(is_fine_owned,
                  is_fine_required,
                  communicator,
                  is_fine_required_ranks,
                  false);

        Utilities::MPI::ConsensusAlgorithms::Selector<
          std::vector<
            std::pair<types::global_cell_index, types::global_cell_index>>,
          std::vector<unsigned int>>
          consensus_algorithm;
        consensus_algorithm.run(process, communicator);

        for (unsigned i = 0; i < is_fine_required.n_elements(); ++i)
          if (is_fine_required_ranks[i] == my_rank)
            ++cells_local[lvl + 1];
          else if (is_fine_required_ranks[i] != numbers::invalid_unsigned_int)
            ++cells_remote[lvl + 1];
      }

    return {cells_local, cells_remote, trias.back()->get_communicator()};
  }

  template <int dim, int spacedim>
  std::tuple<std::vector<types::global_dof_index>,
             std::vector<types::global_dof_index>,
             MPI_Comm>
  horizontal_communication_cost(const Triangulation<dim, spacedim> &tria)
  {
    const unsigned int n_global_levels = tria.n_global_levels();

    std::vector<types::global_dof_index> cells_local(n_global_levels);
    std::vector<types::global_dof_index> cells_remote(n_global_levels);

    for (unsigned int lvl = 0; lvl < n_global_levels; ++lvl)
      for (const auto &cell : tria.cell_iterators_on_level(lvl))
        if (cell->is_locally_owned_on_level())
          ++cells_local[lvl];
        else if (cell->is_ghost_on_level())
          ++cells_remote[lvl];

    return {cells_local, cells_remote, tria.get_communicator()};
  }

  template <int dim, int spacedim>
  std::tuple<std::vector<types::global_dof_index>,
             std::vector<types::global_dof_index>,
             MPI_Comm>
  horizontal_communication_cost(
    const std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
      &trias)
  {
    const unsigned int n_global_levels = trias.size();

    std::vector<types::global_dof_index> cells_local(n_global_levels);
    std::vector<types::global_dof_index> cells_remote(n_global_levels);

    for (unsigned int lvl = 0; lvl < n_global_levels; ++lvl)
      for (const auto &cell : trias[lvl]->active_cell_iterators())
        if (cell->is_locally_owned())
          ++cells_local[lvl];
        else if (cell->is_ghost())
          ++cells_remote[lvl];

    return {cells_local, cells_remote, trias.back()->get_communicator()};
  }

  template <int dim, int spacedim>
  std::tuple<std::vector<double>, MPI_Comm>
  memory_consumption(const Triangulation<dim, spacedim> &tria)
  {
    return {std::vector<double>{static_cast<double>(tria.memory_consumption())},
            tria.get_communicator()};
  }

  template <int dim, int spacedim>
  std::tuple<std::vector<double>, MPI_Comm>
  memory_consumption(
    const std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
      &trias)
  {
    const unsigned int n_global_levels = trias.size();

    std::vector<double> mem(n_global_levels);

    for (unsigned int lvl = 0; lvl < n_global_levels; ++lvl)
      mem[lvl] = static_cast<double>(trias[lvl]->memory_consumption());

    return {mem, trias.back()->get_communicator()};
  }

  template <typename T>
  double
  workload_imbalance(const T &trias)
  {
    const auto [n_cells_on_leves, comm] = workload(trias);
    return internal::workload_imbalance(n_cells_on_leves, comm);
  }

  template <typename T>
  std::vector<std::pair<std::string, double>>
  print_multigrid_statistics(const T &trias)
  {
    std::vector<std::pair<std::string, double>> result;

    // workload
    {
      const auto [n_cells_on_leves, comm] = workload(trias);

      std::vector<types::global_dof_index> n_cells_on_leves_min(
        n_cells_on_leves.size());
      std::vector<types::global_dof_index> n_cells_on_leves_max(
        n_cells_on_leves.size());
      std::vector<types::global_dof_index> n_cells_on_leves_sum(
        n_cells_on_leves.size());

      Utilities::MPI::min(n_cells_on_leves, comm, n_cells_on_leves_min);
      Utilities::MPI::max(n_cells_on_leves, comm, n_cells_on_leves_max);
      Utilities::MPI::sum(n_cells_on_leves, comm, n_cells_on_leves_sum);

      const unsigned int n_proc = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int rank   = Utilities::MPI::this_mpi_process(comm);

      const auto workload_eff =
        1.0 / internal::workload_imbalance(n_cells_on_leves, comm);

      const auto gathered = Utilities::MPI::gather(comm, n_cells_on_leves, 0);

      if (rank == 0)
        {
          std::vector<std::vector<types::global_dof_index>> temp(
            n_cells_on_leves.size(),
            std::vector<types::global_dof_index>((n_proc + 48 - 1) / 48));

          for (unsigned int j = 0; j < n_cells_on_leves.size(); ++j)
            for (unsigned int i = 0; i < n_proc; ++i)
              temp[j][i / 48] += gathered[i][j];

          types::global_dof_index path_max_node = 0;

          for (const auto &i : temp)
            path_max_node += *std::max_element(i.begin(), i.end());

          const double workload_path_max =
            std::accumulate(n_cells_on_leves_max.begin(),
                            n_cells_on_leves_max.end(),
                            0);

          result.emplace_back("workload_eff", workload_eff);
          result.emplace_back("workload_path_max", workload_path_max);

          std::cout << "Workload: " << std::endl;
          std::cout << "  efficiency: " << workload_eff
                    << " max path: " << workload_path_max
                    << " max path node: " << path_max_node << std::endl;

          ConvergenceTable table;

          for (unsigned int i = 0; i < n_cells_on_leves.size(); ++i)
            {
              table.add_value("level", i);
              table.add_value("n_cells_min", n_cells_on_leves_min[i]);
              table.add_value("n_cells_max", n_cells_on_leves_max[i]);
              table.add_value("n_cells_avg", n_cells_on_leves_sum[i] / n_proc);
            }

          table.write_text(std::cout, ConvergenceTable::org_mode_table);

          std::cout << std::endl;
        }
    }

    // vertical communication
    {
      const auto [cells_local, cells_remote, comm] =
        vertical_communication_cost(trias);

      std::vector<types::global_dof_index> cells_local_min(cells_local.size());
      std::vector<types::global_dof_index> cells_local_max(cells_local.size());
      std::vector<types::global_dof_index> cells_local_sum(cells_local.size());

      Utilities::MPI::min(cells_local, comm, cells_local_min);
      Utilities::MPI::max(cells_local, comm, cells_local_max);
      Utilities::MPI::sum(cells_local, comm, cells_local_sum);

      std::vector<types::global_dof_index> cells_remote_min(cells_local.size());
      std::vector<types::global_dof_index> cells_remote_max(cells_local.size());
      std::vector<types::global_dof_index> cells_remote_sum(cells_local.size());

      Utilities::MPI::min(cells_remote, comm, cells_remote_min);
      Utilities::MPI::max(cells_remote, comm, cells_remote_max);
      Utilities::MPI::sum(cells_remote, comm, cells_remote_sum);

      const unsigned int n_proc = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int rank   = Utilities::MPI::this_mpi_process(comm);

      if (rank == 0)
        {
          std::cout << "Vertical communication: " << std::endl;

          const auto n_cells_local =
            std::accumulate(cells_local_sum.begin(), cells_local_sum.end(), 0);
          const auto n_cells_remote = std::accumulate(cells_remote_sum.begin(),
                                                      cells_remote_sum.end(),
                                                      0);

          const double vertical_eff = static_cast<double>(n_cells_local) /
                                      (n_cells_local + n_cells_remote);

          result.emplace_back("vertical_eff", vertical_eff);

          std::cout << "  efficiency: " << vertical_eff << " total: "
                    << std::accumulate(cells_remote_sum.begin(),
                                       cells_remote_sum.end(),
                                       0)
                    << std::endl;

          ConvergenceTable table;

          for (unsigned int i = 0; i < cells_local.size(); ++i)
            {
              table.add_value("level", i);
              table.add_value("n_local_cells_min", cells_local_min[i]);
              table.add_value("n_local_cells_max", cells_local_max[i]);
              table.add_value("n_local_cells_avg", cells_local_sum[i] / n_proc);
              table.add_value("n_remote_cells_min", cells_remote_min[i]);
              table.add_value("n_remote_cells_max", cells_remote_max[i]);
              table.add_value("n_remote_cells_avg",
                              cells_remote_sum[i] / n_proc);
            }

          table.write_text(std::cout, ConvergenceTable::org_mode_table);

          std::cout << std::endl;
        }
    }

    // horizontal communication
    {
      const auto [cells_local, cells_remote, comm] =
        horizontal_communication_cost(trias);

      std::vector<types::global_dof_index> cells_local_min(cells_local.size());
      std::vector<types::global_dof_index> cells_local_max(cells_local.size());
      std::vector<types::global_dof_index> cells_local_sum(cells_local.size());

      Utilities::MPI::min(cells_local, comm, cells_local_min);
      Utilities::MPI::max(cells_local, comm, cells_local_max);
      Utilities::MPI::sum(cells_local, comm, cells_local_sum);

      std::vector<types::global_dof_index> cells_remote_min(cells_local.size());
      std::vector<types::global_dof_index> cells_remote_max(cells_local.size());
      std::vector<types::global_dof_index> cells_remote_sum(cells_local.size());

      Utilities::MPI::min(cells_remote, comm, cells_remote_min);
      Utilities::MPI::max(cells_remote, comm, cells_remote_max);
      Utilities::MPI::sum(cells_remote, comm, cells_remote_sum);

      const unsigned int n_proc = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int rank   = Utilities::MPI::this_mpi_process(comm);

      if (rank == 0)
        {
          std::cout << "Horizontal communication: " << std::endl;

          const auto n_cells_local =
            std::accumulate(cells_local_sum.begin(), cells_local_sum.end(), 0);
          const auto n_cells_remote = std::accumulate(cells_remote_sum.begin(),
                                                      cells_remote_sum.end(),
                                                      0);

          const double horizontal_eff =
            static_cast<double>(n_cells_local + n_cells_remote / 2) /
            (n_cells_local + n_cells_remote);

          result.emplace_back("horizontal_eff", horizontal_eff);

          std::cout << "  efficiency: " << horizontal_eff << " total: "
                    << std::accumulate(cells_remote_sum.begin(),
                                       cells_remote_sum.end(),
                                       0)
                    << std::endl;

          ConvergenceTable table;

          for (unsigned int i = 0; i < cells_local.size(); ++i)
            {
              table.add_value("level", i);
              table.add_value("n_local_cells_min", cells_local_min[i]);
              table.add_value("n_local_cells_max", cells_local_max[i]);
              table.add_value("n_local_cells_avg", cells_local_sum[i] / n_proc);
              table.add_value("n_remote_cells_min", cells_remote_min[i]);
              table.add_value("n_remote_cells_max", cells_remote_max[i]);
              table.add_value("n_remote_cells_avg",
                              cells_remote_sum[i] / n_proc);
            }

          table.write_text(std::cout, ConvergenceTable::org_mode_table);

          std::cout << std::endl;
        }
    }

    // memory consumption
    {
      const auto [mem_local, comm] = memory_consumption(trias);

      std::vector<double> mem_local_min(mem_local.size());
      std::vector<double> mem_local_max(mem_local.size());
      std::vector<double> mem_local_sum(mem_local.size());

      Utilities::MPI::min(mem_local, comm, mem_local_min);
      Utilities::MPI::max(mem_local, comm, mem_local_max);
      Utilities::MPI::sum(mem_local, comm, mem_local_sum);

      const unsigned int n_proc = Utilities::MPI::n_mpi_processes(comm);
      const unsigned int rank   = Utilities::MPI::this_mpi_process(comm);

      if (rank == 0)
        {
          double mem_total =
            std::accumulate(mem_local_sum.begin(), mem_local_sum.end(), 0.0);

          result.emplace_back("mem_total", mem_total);

          std::cout << "Memory consumption: " << std::endl;
          std::cout << "  total: " << mem_total << std::endl;

          ConvergenceTable table;

          for (unsigned int i = 0; i < mem_local.size(); ++i)
            {
              table.add_value("level", i);
              table.add_value("mem_min", mem_local_min[i]);
              table.add_value("mem_max", mem_local_max[i]);
              table.add_value("mem_avg", mem_local_sum[i] / n_proc);
            }

          table.write_text(std::cout, ConvergenceTable::org_mode_table);

          std::cout << std::endl << std::endl << std::endl;
        }
    }
    return result;
  }


  template <int dim>
  std::vector<std::shared_ptr<parallel::distributed::Triangulation<dim>>>
  create_non_nested_sequence(const std::string &geometry_type,
                             const unsigned int n_levels,
                             const MPI_Comm &   mpi_comm)
  {
    Assert(n_levels > 0,
           ExcMessage("A number of levels greater than 0 must be given."));
    // Number of levels is hardcoded here, as hierarchy of grids is given a
    // priori.
    std::string suffix = ".msh"; // gmsh
    if constexpr (dim == 3)
      suffix = ".inp";

    GridIn<dim> grid_in;
    std::vector<std::shared_ptr<parallel::distributed::Triangulation<dim>>>
      trias(n_levels + 1);
    for (unsigned int l = 0; l < trias.size(); ++l)
      {
        trias[l] =
          std::make_shared<parallel::distributed::Triangulation<dim>>(mpi_comm);
        grid_in.attach_triangulation(*trias[l]);

        std::ifstream input_file("../meshes/" + geometry_type + "/" +
                                 geometry_type + "_" + std::to_string(l) +
                                 suffix);
        if constexpr (dim == 2)
          grid_in.read_msh(input_file);
        else
          grid_in.read_abaqus(input_file);
      }
    return trias;
  }
} // namespace dealii::MGTools
