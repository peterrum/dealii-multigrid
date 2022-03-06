

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/grid_generator.h"
#include "include/mg_tools.h"
#include "include/operator.h"
#include "include/scoped_timer.h"

using namespace dealii;

template <int dim>
class GaussianSolution : public dealii::Function<dim>
{
public:
  GaussianSolution(const std::vector<Point<dim>> &source_centers,
                   const double                   width)
    : dealii::Function<dim>()
    , source_centers(source_centers)
    , width(width)
  {}

  double
  value(dealii::Point<dim> const &p, unsigned int const /*component*/ = 0) const
  {
    double return_value = 0;

    for (const auto &source_center : this->source_centers)
      {
        const dealii::Tensor<1, dim> x_minus_xi = p - source_center;
        return_value +=
          std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
      }

    return return_value / dealii::Utilities::fixed_power<dim>(
                            std::sqrt(2. * dealii::numbers::PI) * this->width);
  }

private:
  const std::vector<Point<dim>> source_centers;
  const double                  width;
};

template <int dim>
class GaussianRightHandSide : public dealii::Function<dim>
{
public:
  GaussianRightHandSide(const std::vector<Point<dim>> &source_centers,
                        const double                   width)
    : dealii::Function<dim>()
    , source_centers(source_centers)
    , width(width)
  {}

  double
  value(dealii::Point<dim> const &p, unsigned int const /*component*/ = 0) const
  {
    double const coef         = 1.0;
    double       return_value = 0;

    for (const auto &source_center : this->source_centers)
      {
        const dealii::Tensor<1, dim> x_minus_xi = p - source_center;

        return_value +=
          ((2 * dim * coef -
            4 * coef * x_minus_xi.norm_square() / (this->width * this->width)) /
           (this->width * this->width) *
           std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
      }

    return return_value / dealii::Utilities::fixed_power<dim>(
                            std::sqrt(2 * dealii::numbers::PI) * this->width);
  }

private:
  const std::vector<Point<dim>> source_centers;
  const double                  width;
};

namespace dealii::parallel
{
  template <int dim, int spacedim = dim>
  class Helper
  {
  public:
    Helper(Triangulation<dim, spacedim> &triangulation)
    {
      reinit(triangulation);

      const auto fu = [&]() { this->reinit(triangulation); };

      triangulation.signals.post_p4est_refinement.connect(fu);
      triangulation.signals.post_distributed_refinement.connect(fu);
      triangulation.signals.pre_distributed_repartition.connect(fu);
      triangulation.signals.post_distributed_repartition.connect(fu);
    }

    void
    reinit(const Triangulation<dim, spacedim> &triangulation)
    {
      if (dim == 3)
        {
          this->line_to_cells.clear();

          const unsigned int n_raw_lines = triangulation.n_raw_lines();
          this->line_to_cells.resize(n_raw_lines);

          // In 3D, we can have DoFs on only an edge being constrained (e.g. in
          // a cartesian 2x2x2 grid, where only the upper left 2 cells are
          // refined). This sets up a helper data structure in the form of a
          // mapping from edges (i.e. lines) to neighboring cells.

          // Mapping from an edge to which children that share that edge.
          const unsigned int line_to_children[12][2] = {{0, 2},
                                                        {1, 3},
                                                        {0, 1},
                                                        {2, 3},
                                                        {4, 6},
                                                        {5, 7},
                                                        {4, 5},
                                                        {6, 7},
                                                        {0, 4},
                                                        {1, 5},
                                                        {2, 6},
                                                        {3, 7}};

          std::vector<std::vector<
            std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                      unsigned int>>>
            line_to_inactive_cells(n_raw_lines);

          // First add active and inactive cells to their lines:
          for (const auto &cell : triangulation.cell_iterators())
            {
              for (unsigned int line = 0;
                   line < GeometryInfo<3>::lines_per_cell;
                   ++line)
                {
                  const unsigned int line_idx = cell->line(line)->index();
                  if (cell->is_active())
                    line_to_cells[line_idx].push_back(
                      std::make_pair(cell, line));
                  else
                    line_to_inactive_cells[line_idx].push_back(
                      std::make_pair(cell, line));
                }
            }

          // Now, we can access edge-neighboring active cells on same level to
          // also access of an edge to the edges "children". These are found
          // from looking at the corresponding edge of children of inactive edge
          // neighbors.
          for (unsigned int line_idx = 0; line_idx < n_raw_lines; ++line_idx)
            {
              if ((line_to_cells[line_idx].size() > 0) &&
                  line_to_inactive_cells[line_idx].size() > 0)
                {
                  // We now have cells to add (active ones) and edges to which
                  // they should be added (inactive cells).
                  const auto &inactive_cell =
                    line_to_inactive_cells[line_idx][0].first;
                  const unsigned int neighbor_line =
                    line_to_inactive_cells[line_idx][0].second;

                  for (unsigned int c = 0; c < 2; ++c)
                    {
                      const auto &child = inactive_cell->child(
                        line_to_children[neighbor_line][c]);
                      const unsigned int child_line_idx =
                        child->line(neighbor_line)->index();

                      // Now add all active cells
                      for (const auto &cl : line_to_cells[line_idx])
                        line_to_cells[child_line_idx].push_back(cl);
                    }
                }
            }
        }
    }


    bool
    is_constrained(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return is_face_constrained(cell) || is_edge_constrained(cell);
    }

    bool
    is_face_constrained(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      if (cell->is_locally_owned())
        for (unsigned int f : cell->face_indices())
          if (!cell->at_boundary(f) &&
              (cell->level() > cell->neighbor(f)->level()))
            return true;

      return false;
    }

    bool
    is_edge_constrained(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      if (dim == 3)
        if (cell->is_locally_owned())
          for (const auto line : cell->line_indices())
            for (const auto &other_cell :
                 line_to_cells[cell->line(line)->index()])
              if (cell->level() > other_cell.first->level())
                return true;

      return false;
    }

  private:
    std::vector<std::vector<
      std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                unsigned int>>>
      line_to_cells;
  };

  template <int dim, int spacedim = dim>
  std::function<
    unsigned int(const typename Triangulation<dim, spacedim>::cell_iterator &,
                 const typename Triangulation<dim, spacedim>::CellStatus)>
  hanging_nodes_weighting(const Helper<dim, spacedim> &helper,
                          const double                 weight)
  {
    return [&helper, weight](const auto &cell, const auto &) -> unsigned int {
      if (cell->is_active() == false || cell->is_locally_owned() == false)
        return 10000;

      if (helper.is_constrained(cell))
        return 10000 * weight;
      else
        return 10000;
    };
  }


} // namespace dealii::parallel

struct MultigridParameters
{
  struct
  {
    std::string  type            = "amg"; // "cg";
    unsigned int maxiter         = 10000;
    double       abstol          = 1e-20;
    double       reltol          = 1e-4;
    unsigned int smoother_sweeps = 1;
    unsigned int n_cycles        = 1;
    std::string  smoother_type   = "ILU";
  } coarse_solver;

  struct
  {
    std::string  type                = "chebyshev";
    double       smoothing_range     = 20;
    unsigned int degree              = 5;
    unsigned int eig_cg_n_iterations = 20;
  } smoother;

  struct
  {
    unsigned int maxiter = 10000;
    double       abstol  = 1e-20;
    double       reltol  = 1e-4;
  } cg_normal;

  struct
  {
    unsigned int maxiter = 20;
    double       abstol  = 1e-40;
    double       reltol  = 1e-40;
  } cg_parameter_study;

  bool         do_parameter_study = false;
  unsigned int n_repetitions      = 5;
};


void
monitor(const std::string &label)
{
  return;

  dealii::Utilities::System::MemoryStats stats;
  dealii::Utilities::System::get_memory_stats(stats);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "MONITOR " << label << ": ";

  if (label != "break")
    {
      const auto print = [&pcout](const double value) {
        const auto min_max_avg =
          dealii::Utilities::MPI::min_max_avg(value / 1e6, MPI_COMM_WORLD);

        pcout << min_max_avg.min << " " << min_max_avg.max << " "
              << min_max_avg.avg << " " << min_max_avg.sum << " ";
      };

      print(stats.VmPeak);
      print(stats.VmSize);
      print(stats.VmHWM);
      print(stats.VmRSS);
    }

  pcout << std::endl;
}

namespace dealii
{
  namespace RepartitioningPolicyTools
  {
    /**
     * A class to use for the deal.II coarsening functionality, where we try to
     * balance the mesh coarsening with a minimum granularity and the number of
     * partitions on coarser levels.
     */
    template <int dim, int spacedim = dim>
    class BalancedGranularityPartitionPolicy
      : public RepartitioningPolicyTools::Base<dim, spacedim>
    {
    public:
      BalancedGranularityPartitionPolicy(unsigned int const n_mpi_processes)
        : n_mpi_processes_per_level{n_mpi_processes}
      {}

      virtual LinearAlgebra::distributed::Vector<double>
      partition(
        Triangulation<dim, spacedim> const &tria_coarse_in) const override
      {
        types::global_cell_index const n_cells =
          tria_coarse_in.n_global_active_cells();

        // TODO: We hard-code a grain-size limit of 200 cells per processor
        // (assuming linear finite elements and typical behavior of
        // supercomputers). In case we have fewer cells on the fine level, we do
        // not immediately go to 200 cells per rank, but limit the growth by a
        // factor of 8, which limits makes sure that we do not create too many
        // messages for individual MPI processes.
        unsigned int const grain_size_limit = std::min<unsigned int>(
          200U, 8 * n_cells / n_mpi_processes_per_level.back() + 1);

        RepartitioningPolicyTools::MinimalGranularityPolicy<dim, spacedim>
          partitioning_policy(grain_size_limit);
        LinearAlgebra::distributed::Vector<double> const partitions =
          partitioning_policy.partition(tria_coarse_in);

        // The vector 'partitions' contains the partition numbers. To get the
        // number of partitions, we take the infinity norm.
        n_mpi_processes_per_level.push_back(
          static_cast<unsigned int>(partitions.linfty_norm()) + 1);
        return partitions;
      }

    private:
      mutable std::vector<unsigned int> n_mpi_processes_per_level;
    };
  } // namespace RepartitioningPolicyTools
} // namespace dealii

namespace dealii
{
  /**
   * Coarse grid solver using a preconditioner only. This is a little wrapper,
   * transforming a preconditioner into a coarse grid solver.
   */
  template <class VectorType, class PreconditionerType>
  class MGCoarseGridApplyPreconditioner : public MGCoarseGridBase<VectorType>
  {
  public:
    /**
     * Default constructor.
     */
    MGCoarseGridApplyPreconditioner();

    /**
     * Constructor. Store a pointer to the preconditioner for later use.
     */
    MGCoarseGridApplyPreconditioner(const PreconditionerType &precondition);

    /**
     * Clear the pointer.
     */
    void
    clear();

    /**
     * Initialize new data.
     */
    void
    initialize(const PreconditionerType &precondition);

    /**
     * Implementation of the abstract function.
     */
    virtual void
    operator()(const unsigned int level,
               VectorType &       dst,
               const VectorType & src) const override;

  private:
    /**
     * Reference to the preconditioner.
     */
    SmartPointer<
      const PreconditionerType,
      MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>>
      preconditioner;
  };



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner()
    : preconditioner(0, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner(const PreconditionerType &preconditioner)
    : preconditioner(&preconditioner, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::initialize(
    const PreconditionerType &preconditioner_)
  {
    preconditioner = &preconditioner_;
  }



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::clear()
  {
    preconditioner = 0;
  }


  namespace internal
  {
    namespace MGCoarseGridApplyPreconditioner
    {
      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner,
            VectorType &             dst,
            const VectorType &       src)
      {
        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst, src);
      }

      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  !std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner,
            VectorType &             dst,
            const VectorType &       src)
      {
        LinearAlgebra::distributed::Vector<double> src_;
        LinearAlgebra::distributed::Vector<double> dst_;

        src_ = src;
        dst_ = dst;

        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst_, src_);

        dst = dst_;
      }
    } // namespace MGCoarseGridApplyPreconditioner
  }   // namespace internal


  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::operator()(
    const unsigned int /*level*/,
    VectorType &      dst,
    const VectorType &src) const
  {
    internal::MGCoarseGridApplyPreconditioner::solve(preconditioner, dst, src);
  }
} // namespace dealii

namespace dealii
{
#ifdef DEAL_II_WITH_PETSC

  typedef Vec VectorTypePETSc;

  /*
   *  This function wraps the copy of a PETSc object (sparse matrix,
   *  preconditioner) with a dealii::LinearAlgebra::distributed::Vector, taking
   *  pre-allocated PETSc vector objects (with struct name `Vec`, aka
   *  VectorTypePETSc) for the temporary operations
   */
  template <typename VectorType>
  void
  apply_petsc_operation(
    VectorType &                                           dst,
    VectorType const &                                     src,
    VectorTypePETSc &                                      petsc_vector_dst,
    VectorTypePETSc &                                      petsc_vector_src,
    std::function<void(PETScWrappers::VectorBase &,
                       PETScWrappers::VectorBase const &)> petsc_operation)
  {
    {
      // copy to PETSc internal vector type because there is currently no such
      // function in deal.II (and the transition via ReadWriteVector is too
      // slow/poorly tested)
      PetscInt       begin, end;
      PetscErrorCode ierr =
        VecGetOwnershipRange(petsc_vector_src, &begin, &end);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      PetscScalar *ptr;
      ierr = VecGetArray(petsc_vector_src, &ptr);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      const PetscInt local_size = src.get_partitioner()->locally_owned_size();
      AssertDimension(local_size, static_cast<unsigned int>(end - begin));
      for (PetscInt i = 0; i < local_size; ++i)
        {
          ptr[i] = src.local_element(i);
        }

      ierr = VecRestoreArray(petsc_vector_src, &ptr);
      AssertThrow(ierr == 0, ExcPETScError(ierr));
    }

    // wrap `Vec` (aka VectorTypePETSc) into VectorBase (without copying data)
    PETScWrappers::VectorBase petsc_dst(petsc_vector_dst);
    PETScWrappers::VectorBase petsc_src(petsc_vector_src);

    petsc_operation(petsc_dst, petsc_src);

    {
      PetscInt       begin, end;
      PetscErrorCode ierr =
        VecGetOwnershipRange(petsc_vector_dst, &begin, &end);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      PetscScalar *ptr;
      ierr = VecGetArray(petsc_vector_dst, &ptr);
      AssertThrow(ierr == 0, ExcPETScError(ierr));

      const PetscInt local_size = dst.get_partitioner()->locally_owned_size();
      AssertDimension(local_size, static_cast<unsigned int>(end - begin));

      for (PetscInt i = 0; i < local_size; ++i)
        {
          dst.local_element(i) = ptr[i];
        }

      ierr = VecRestoreArray(petsc_vector_dst, &ptr);
      AssertThrow(ierr == 0, ExcPETScError(ierr));
    }
  }

  /*
   *  This function wraps the copy of a PETSc object (sparse matrix,
   *  preconditioner) with a dealii::LinearAlgebra::distributed::Vector,
   *  allocating a PETSc vectors and then calling the other function
   */
  template <typename VectorType>
  void
  apply_petsc_operation(
    VectorType &      dst,
    VectorType const &src,
    MPI_Comm const &  petsc_mpi_communicator,
    std::function<void(PETScWrappers::VectorBase &,
                       PETScWrappers::VectorBase const &)> petsc_operation)
  {
    VectorTypePETSc petsc_vector_dst, petsc_vector_src;
    VecCreateMPI(petsc_mpi_communicator,
                 dst.get_partitioner()->locally_owned_size(),
                 PETSC_DETERMINE,
                 &petsc_vector_dst);
    VecCreateMPI(petsc_mpi_communicator,
                 src.get_partitioner()->locally_owned_size(),
                 PETSC_DETERMINE,
                 &petsc_vector_src);

    apply_petsc_operation(
      dst, src, petsc_vector_dst, petsc_vector_src, petsc_operation);

    PetscErrorCode ierr = VecDestroy(&petsc_vector_dst);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    ierr = VecDestroy(&petsc_vector_src);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }

  /**
   * Coarse grid solver using a preconditioner only. This is a little wrapper,
   * transforming a preconditioner into a coarse grid solver.
   */
  template <class VectorType, class PreconditionerType>
  class MGCoarseGridApplyPreconditionerPETSC
    : public MGCoarseGridBase<VectorType>
  {
    using VectorTypePETSc = Vec;

  public:
    /**
     * Default constructor.
     */
    MGCoarseGridApplyPreconditionerPETSC();

    /**
     * Constructor. Store a pointer to the preconditioner for later use.
     */
    MGCoarseGridApplyPreconditionerPETSC(const PreconditionerType &precondition,
                                         const MPI_Comm     mpi_communicator,
                                         const unsigned int locally_owned_size);

    ~MGCoarseGridApplyPreconditionerPETSC()
    {
      if (preconditioner)
        {
          PetscErrorCode ierr = VecDestroy(&petsc_vector_dst);
          AssertThrow(ierr == 0, ExcPETScError(ierr));
          ierr = VecDestroy(&petsc_vector_src);
          AssertThrow(ierr == 0, ExcPETScError(ierr));
        }
    }

    /**
     * Clear the pointer.
     */
    void
    clear();

    /**
     * Initialize new data.
     */
    void
    initialize(const PreconditionerType &precondition,
               const MPI_Comm            mpi_communicator,
               const unsigned int        locally_owned_size);

    /**
     * Implementation of the abstract function.
     */
    virtual void
    operator()(const unsigned int level,
               VectorType &       dst,
               const VectorType & src) const override;

  private:
    /**
     * Reference to the preconditioner.
     */
    SmartPointer<
      const PreconditionerType,
      MGCoarseGridApplyPreconditionerPETSC<VectorType, PreconditionerType>>
      preconditioner;

    mutable VectorTypePETSc petsc_vector_src;
    mutable VectorTypePETSc petsc_vector_dst;
  };



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditionerPETSC<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditionerPETSC()
    : preconditioner(0, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditionerPETSC<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditionerPETSC(
      const PreconditionerType &preconditioner,
      const MPI_Comm            mpi_communicator,
      const unsigned int        locally_owned_size)
  {
    initialize(preconditioner, mpi_communicator, locally_owned_size);
  }



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditionerPETSC<VectorType, PreconditionerType>::
    initialize(const PreconditionerType &preconditioner_,
               const MPI_Comm            mpi_communicator,
               const unsigned int        locally_owned_size)
  {
    preconditioner = &preconditioner_;

    VecCreateMPI(mpi_communicator,
                 locally_owned_size,
                 PETSC_DETERMINE,
                 &petsc_vector_dst);
    VecCreateMPI(mpi_communicator,
                 locally_owned_size,
                 PETSC_DETERMINE,
                 &petsc_vector_src);
  }



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditionerPETSC<VectorType, PreconditionerType>::clear()
  {
    preconditioner = 0;
  }



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditionerPETSC<VectorType, PreconditionerType>::
  operator()(const unsigned int /*level*/,
             VectorType &      dst,
             const VectorType &src) const
  {
    if (preconditioner != nullptr)
      apply_petsc_operation(dst,
                            src,
                            petsc_vector_dst,
                            petsc_vector_src,
                            [&](auto &dst, const auto &src) {
                              if (preconditioner != nullptr)
                                preconditioner->vmult(dst, src);
                            });
  }
#endif
} // namespace dealii



template <int dim,
          typename SystemMatrixType,
          typename LevelMatrixType,
          typename MGTransferTypeFine,
          typename MGTransferTypeCoarse>
static void
mg_solve(SolverControl &                              solver_control,
         typename SystemMatrixType::VectorType &      dst,
         const typename SystemMatrixType::VectorType &src,
         const MultigridParameters &                  mg_data,
         const DoFHandler<dim> &                      dof_fine,
         const DoFHandler<dim> &                      dof_intermediate,
         const SystemMatrixType &                     fine_matrix,
         const MGLevelObject<LevelMatrixType> &       mg_matrices,
         const MGTransferTypeFine &                   mg_transfer_fine,
         const MGTransferTypeCoarse &                 mg_transfer_intermediate,
         const unsigned int                           offset,
         const bool                                   verbose,
         const MPI_Comm &                             sub_comm,
         ConvergenceTable &                           table)
{
  AssertThrow(mg_data.smoother.type == "chebyshev", ExcNotImplemented());

  (void)sub_comm;

  monitor("mg_solve::0");

  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  using VectorType                 = typename LevelMatrixType::VectorType;
  using Number                     = typename LevelMatrixType::value_type;
  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                             VectorType,
                                             SmootherPreconditionerType>;

  // Initialize level operators.
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  // Initialize smoothers.
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level,
                                                                     max_level);

  for (unsigned int level = min_level; level <= max_level; level++)
    {
      smoother_data[level].preconditioner =
        std::make_shared<SmootherPreconditionerType>();
      mg_matrices[level].compute_inverse_diagonal(
        smoother_data[level].preconditioner->get_vector());
      smoother_data[level].smoothing_range = mg_data.smoother.smoothing_range;
      smoother_data[level].degree          = mg_data.smoother.degree;
      smoother_data[level].eig_cg_n_iterations =
        mg_data.smoother.eig_cg_n_iterations;
    }

  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  mg_smoother.initialize(mg_matrices, smoother_data);

  monitor("mg_solve::1");

  // Initialize coarse-grid solver.
  ReductionControl     coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                              mg_data.coarse_solver.abstol,
                                              mg_data.coarse_solver.reltol,
                                              false,
                                              false);
  SolverCG<VectorType> coarse_grid_solver(coarse_grid_solver_control);
  SolverCG<typename SystemMatrixType::VectorType> coarse_grid_solver_(
    coarse_grid_solver_control);

  PreconditionIdentity precondition_identity;
  PreconditionChebyshev<LevelMatrixType, VectorType, DiagonalMatrix<VectorType>>
    precondition_chebyshev;

#ifdef DEAL_II_WITH_TRILINOS
  TrilinosWrappers::PreconditionAMG precondition_amg;
#endif

#ifdef DEAL_II_WITH_PETSC
  PETScWrappers::PreconditionBoomerAMG precondition_amg_petsc;
#endif

  std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  if (mg_data.coarse_solver.type == "cg")
    {
      // CG with identity matrix as preconditioner

      mg_coarse =
        std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                     SolverCG<VectorType>,
                                                     LevelMatrixType,
                                                     PreconditionIdentity>>(
          coarse_grid_solver, mg_matrices[min_level], precondition_identity);
    }
  else if (mg_data.coarse_solver.type == "cg_with_chebyshev")
    {
      // CG with Chebyshev as preconditioner

      typename SmootherType::AdditionalData smoother_data;

      smoother_data.preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>();
      mg_matrices[min_level].compute_inverse_diagonal(
        smoother_data.preconditioner->get_vector());
      smoother_data.smoothing_range     = mg_data.smoother.smoothing_range;
      smoother_data.degree              = mg_data.smoother.degree;
      smoother_data.eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;

      precondition_chebyshev.initialize(mg_matrices[min_level], smoother_data);

      mg_coarse = std::make_unique<
        MGCoarseGridIterativeSolver<VectorType,
                                    SolverCG<VectorType>,
                                    LevelMatrixType,
                                    decltype(precondition_chebyshev)>>(
        coarse_grid_solver, mg_matrices[min_level], precondition_chebyshev);
    }
  else if (mg_data.coarse_solver.type == "cg_with_amg")
    {
      // CG with AMG as preconditioner

#ifdef DEAL_II_WITH_TRILINOS
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
      amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
      amg_data.smoother_type   = mg_data.coarse_solver.smoother_type.c_str();

      // CG with AMG as preconditioner
      precondition_amg.initialize(
        mg_matrices[min_level].get_trilinos_system_matrix(), amg_data);

      mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<
        VectorType,
        SolverCG<typename SystemMatrixType::VectorType>,
        TrilinosWrappers::SparseMatrix,
        decltype(precondition_amg)>>(
        coarse_grid_solver_,
        mg_matrices[min_level].get_trilinos_system_matrix(),
        precondition_amg);
#else
      AssertThrow(false, ExcNotImplemented());
#endif
    }
  else if (mg_data.coarse_solver.type == "amg")
    {
      // AMG as coarse-grid solver

#ifdef DEAL_II_WITH_TRILINOS

      // CG with AMG as preconditioner
      if (sub_comm != MPI_COMM_NULL)
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
          amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
          amg_data.smoother_type  = mg_data.coarse_solver.smoother_type.c_str();
          amg_data.output_details = true;

          Teuchos::ParameterList              parameter_list;
          std::unique_ptr<Epetra_MultiVector> distributed_constant_modes;
          amg_data.set_parameters(
            parameter_list,
            distributed_constant_modes,
            mg_matrices[min_level].get_trilinos_system_matrix(sub_comm));
          parameter_list.set("repartition: enable", 1);
          parameter_list.set("repartition: max min ratio", 1.3);
          parameter_list.set("repartition: min per proc", 300);
          parameter_list.set("repartition: partitioner", "Zoltan");
          parameter_list.set("repartition: Zoltan dimensions", 3);

          precondition_amg.initialize(
            mg_matrices[min_level].get_trilinos_system_matrix(sub_comm),
            parameter_list);
          mg_coarse = std::make_unique<
            MGCoarseGridApplyPreconditioner<VectorType,
                                            decltype(precondition_amg)>>(
            precondition_amg);
        }
      else
        {
          mg_coarse = std::make_unique<
            MGCoarseGridApplyPreconditioner<VectorType,
                                            decltype(precondition_amg)>>();
        }

#else
      AssertThrow(false, ExcNotImplemented());
#endif
    }
  else if (mg_data.coarse_solver.type == "amg_petsc")
    {
      // AMG as coarse-grid solver

#ifdef DEAL_II_WITH_PETSC
      PETScWrappers::PreconditionBoomerAMG::AdditionalData amg_data;

#  if false
    amg_data.n_sweeps_coarse = mg_data.coarse_solver.smoother_sweeps;
    amg_data.max_iter        = mg_data.coarse_solver.n_cycles;
    amg_data.relaxation_type_down =
      PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
    amg_data.relaxation_type_up =
      PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
    amg_data.relaxation_type_coarse =
      PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
#  else
      amg_data.strong_threshold                 = 0.5;
      amg_data.aggressive_coarsening_num_levels = 2;
      amg_data.output_details                   = true;
      amg_data.relaxation_type_up = PETScWrappers::PreconditionBoomerAMG::
        AdditionalData::RelaxationType::symmetricSORJacobi;
      amg_data.relaxation_type_down = PETScWrappers::PreconditionBoomerAMG::
        AdditionalData::RelaxationType::symmetricSORJacobi;
      amg_data.relaxation_type_coarse = PETScWrappers::PreconditionBoomerAMG::
        AdditionalData::RelaxationType::GaussianElimination;
      amg_data.n_sweeps_coarse = mg_data.coarse_solver.smoother_sweeps;
      amg_data.max_iter        = mg_data.coarse_solver.n_cycles;
      amg_data.w_cycle         = false;
#  endif

      // CG with AMG as preconditioner
      if (sub_comm != MPI_COMM_NULL)
        {
          VectorType temp;
          mg_matrices[min_level].initialize_dof_vector(temp);

          precondition_amg_petsc.initialize(
            mg_matrices[min_level].get_petsc_system_matrix(sub_comm), amg_data);
          mg_coarse = std::make_unique<MGCoarseGridApplyPreconditionerPETSC<
            VectorType,
            decltype(precondition_amg_petsc)>>(
            precondition_amg_petsc,
            sub_comm,
            temp.get_partitioner()->locally_owned_size());
        }
      else
        {
          mg_coarse = std::make_unique<MGCoarseGridApplyPreconditionerPETSC<
            VectorType,
            decltype(precondition_amg_petsc)>>();
        }

#else
      AssertThrow(false, ExcNotImplemented());
#endif
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  monitor("mg_solve::2");

  // 4) create interface matrices (for local smoothing)
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
    mg_interface_matrices;
  mg_interface_matrices.resize(min_level, max_level);
  for (unsigned int level = min_level; level <= max_level; level++)
    mg_interface_matrices[level].initialize(mg_matrices[level]);
  mg::Matrix<VectorType> mg_interface(mg_interface_matrices);

  monitor("mg_solve::3");

  // Create multigrid object.

  Multigrid<VectorType> mg_intermediate(mg_matrix,
                                        *mg_coarse,
                                        mg_transfer_intermediate,
                                        mg_smoother,
                                        mg_smoother,
                                        min_level,
                                        min_level + offset - 1);

  if constexpr (!std::is_same<
                  MGTransferTypeCoarse,
                  MGTransferGlobalCoarsening<dim, VectorType>>::value)
    if (dof_fine.get_triangulation().has_hanging_nodes())
      mg_intermediate.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, VectorType, MGTransferTypeCoarse> preconditioner_mg(
    dof_intermediate, mg_intermediate, mg_transfer_intermediate);

  std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse_intermediate =
    std::make_unique<MGCoarseGridApplyPreconditioner<
      VectorType,
      PreconditionMG<dim, VectorType, MGTransferTypeCoarse>>>(
      preconditioner_mg);


  Multigrid<VectorType> mg_fine(mg_matrix,
                                offset == 0 ? *mg_coarse :
                                              *mg_coarse_intermediate,
                                mg_transfer_fine,
                                mg_smoother,
                                mg_smoother,
                                min_level + offset,
                                max_level);

  if constexpr (!std::is_same<
                  MGTransferTypeFine,
                  MGTransferGlobalCoarsening<dim, VectorType>>::value)
    if (dof_fine.get_triangulation().has_hanging_nodes())
      mg_fine.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, VectorType, MGTransferTypeFine> preconditioner(
    dof_fine, mg_fine, mg_transfer_fine);

  monitor("mg_solve::4");

  MPI_Barrier(MPI_COMM_WORLD);

  // Finally, solve.
  try
    {
      dst = 0.0;
      SolverCG<typename SystemMatrixType::VectorType>(solver_control)
        .solve(fine_matrix, dst, src, preconditioner);
    }
  catch (const SolverControl::NoConvergence &)
    {}

  const unsigned int n_repetitions = mg_data.n_repetitions;
  unsigned int       counter       = 0;

  std::vector<std::vector<std::vector<
    std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>>
    all_mg_timers(n_repetitions);

  for (unsigned int r = 0; r < n_repetitions; ++r)
    {
      all_mg_timers[r].resize((max_level - min_level + 1));
      for (unsigned int i = 0; i < all_mg_timers[r].size(); ++i)
        all_mg_timers[r][i].resize(7);
    }

  const auto create_mg_timer_function = [&](const unsigned int i,
                                            const std::string &label) {
    return [i, label, &all_mg_timers, &counter](const bool         flag,
                                                const unsigned int level) {
      if (false && flag)
        std::cout << label << " " << level << std::endl;
      if (flag)
        all_mg_timers[counter][level][i].second =
          std::chrono::system_clock::now();
      else
        all_mg_timers[counter][level][i].first +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() -
            all_mg_timers[counter][level][i].second)
            .count() /
          1e9;
    };
  };

  {
    mg_fine.connect_pre_smoother_step(
      create_mg_timer_function(0, "pre_smoother_step"));
    mg_fine.connect_residual_step(create_mg_timer_function(1, "residual_step"));
    mg_fine.connect_restriction(create_mg_timer_function(2, "restriction"));
    mg_fine.connect_coarse_solve(create_mg_timer_function(3, "coarse_solve"));
    mg_fine.connect_prolongation(create_mg_timer_function(4, "prolongation"));
    mg_fine.connect_edge_prolongation(
      create_mg_timer_function(5, "edge_prolongation"));
    mg_fine.connect_post_smoother_step(
      create_mg_timer_function(6, "post_smoother_step"));
  }
  {
    mg_intermediate.connect_pre_smoother_step(
      create_mg_timer_function(0, "pre_smoother_step"));
    mg_intermediate.connect_residual_step(
      create_mg_timer_function(1, "residual_step"));
    mg_intermediate.connect_restriction(
      create_mg_timer_function(2, "restriction"));
    mg_intermediate.connect_coarse_solve(
      create_mg_timer_function(3, "coarse_solve"));
    mg_intermediate.connect_prolongation(
      create_mg_timer_function(4, "prolongation"));
    mg_intermediate.connect_edge_prolongation(
      create_mg_timer_function(5, "edge_prolongation"));
    mg_intermediate.connect_post_smoother_step(
      create_mg_timer_function(6, "post_smoother_step"));
  }

  std::vector<std::vector<
    std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
    all_mg_precon_timers(n_repetitions);

  for (unsigned int i = 0; i < n_repetitions; ++i)
    all_mg_precon_timers[i].resize(2);

  const auto create_mg_precon_timer_function = [&](const unsigned int i) {
    return [i, &all_mg_precon_timers, &counter](const bool flag) {
      if (flag)
        all_mg_precon_timers[counter][i].second =
          std::chrono::system_clock::now();
      else
        all_mg_precon_timers[counter][i].first +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() -
            all_mg_precon_timers[counter][i].second)
            .count() /
          1e9;
    };
  };

  preconditioner.connect_transfer_to_mg(create_mg_precon_timer_function(0));
  preconditioner.connect_transfer_to_global(create_mg_precon_timer_function(1));

  std::vector<double> times(n_repetitions);

  for (; counter < n_repetitions; ++counter)
    {
      MPI_Barrier(MPI_COMM_WORLD);

      double time = 0.0;
      try
        {
          dst = 0.0;
          ScopedTimer timer(time);
          SolverCG<typename SystemMatrixType::VectorType>(solver_control)
            .solve(fine_matrix, dst, src, preconditioner);
        }
      catch (const SolverControl::NoConvergence &)
        {}

      times[counter] = time;
    }

  const auto min_max_avg = Utilities::MPI::min_max_avg(times, MPI_COMM_WORLD);

  const unsigned int min_index =
    std::distance(min_max_avg.begin(),
                  std::min_element(min_max_avg.begin(),
                                   min_max_avg.end(),
                                   [](const auto &a, const auto &b) {
                                     return a.avg < b.avg;
                                   }));

  const auto &time             = times[min_index];
  const auto &mg_timers        = all_mg_timers[min_index];
  const auto &mg_precon_timers = all_mg_precon_timers[min_index];

  double time_cg = time;

  for (unsigned int i = 0; i < mg_timers.size(); ++i)
    for (unsigned int j = 0; j < mg_timers[0].size(); ++j)
      time_cg -= mg_timers[i][j].first;

  for (unsigned int i = 0; i < mg_precon_timers.size(); ++i)
    time_cg -= mg_precon_timers[i].first;
  table.add_value("n_levels", max_level - min_level + 1);
  table.add_value("n_iterations", solver_control.last_step());
  table.add_value("time", time);
  table.add_value("time_cg", time_cg / solver_control.last_step());
  table.add_value("throughput", src.size() * solver_control.last_step() / time);
  table.set_scientific("throughput", true);

  FullMatrix<double> mg_times(mg_timers.size() + 1, mg_timers[0].size() + 1);
  FullMatrix<double> mg_times_incl(mg_timers.size() + 1,
                                   mg_timers[0].size() + 1);

  for (unsigned int i = 0; i < mg_timers.size(); ++i)
    for (unsigned int j = 0; j < mg_timers[0].size(); ++j)
      mg_times[1 + i][j + 1] =
        mg_timers[i][j].first / solver_control.last_step();

  for (unsigned int i = 1; i < mg_timers.size() + 1; ++i)
    for (unsigned int j = 1; j < mg_timers[0].size() + 1; ++j)
      mg_times_incl[i][j] = mg_times_incl[i - 1][j] + mg_times[i][j];

  for (unsigned int i = 0; i < mg_timers.size(); ++i)
    for (unsigned int j = 0; j < mg_timers[0].size(); ++j)
      {
        mg_times[0][j + 1] += mg_times[1 + i][j + 1];
        mg_times[1 + i][0] += mg_times[1 + i][j + 1];
      }

  FullMatrix<double> mg_times_min(mg_times.m(), mg_times.n());
  FullMatrix<double> mg_times_max(mg_times.m(), mg_times.n());
  FullMatrix<double> mg_times_avg(mg_times.m(), mg_times.n());
  {
    const ArrayView<const double> values = make_array_view(mg_times);

    const ArrayView<double> values_min = make_array_view(mg_times_min);
    Utilities::MPI::min<double>(values, MPI_COMM_WORLD, values_min);

    const ArrayView<double> values_max = make_array_view(mg_times_max);
    Utilities::MPI::max<double>(values, MPI_COMM_WORLD, values_max);

    const ArrayView<double> values_avg = make_array_view(mg_times_avg);
    Utilities::MPI::sum<double>(values, MPI_COMM_WORLD, values_avg);
  }

  FullMatrix<double> mg_times_incl_min(mg_times_incl.m(), mg_times_incl.n());
  FullMatrix<double> mg_times_incl_max(mg_times_incl.m(), mg_times_incl.n());
  FullMatrix<double> mg_times_incl_avg(mg_times_incl.m(), mg_times_incl.n());
  {
    const ArrayView<const double> values = make_array_view(mg_times_incl);

    const ArrayView<double> values_min = make_array_view(mg_times_incl_min);
    Utilities::MPI::min<double>(values, MPI_COMM_WORLD, values_min);

    const ArrayView<double> values_max = make_array_view(mg_times_incl_max);
    Utilities::MPI::max<double>(values, MPI_COMM_WORLD, values_max);

    const ArrayView<double> values_avg = make_array_view(mg_times_incl_avg);
    Utilities::MPI::sum<double>(values, MPI_COMM_WORLD, values_avg);
  }

  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  for (unsigned int i = 0; i < mg_times.m(); ++i)
    for (unsigned int j = 0; j < mg_times.n(); ++j)
      {
        mg_times_avg(i, j) /= n_procs;
        mg_times_incl_avg(i, j) /= n_procs;
      }

  if (verbose &&
      Utilities::MPI::this_mpi_process(dof_fine.get_communicator()) == 0)
    {
      std::cout << "RANK 0:" << std::endl;
      mg_times.print_formatted(std::cout, 10, false, 16, "0.0000000000");
      std::cout << std::endl;

      std::cout << "MIN:" << std::endl;
      mg_times_min.print_formatted(std::cout, 10, false, 16, "0.0000000000");
      std::cout << std::endl;

      std::cout << "MAX:" << std::endl;
      mg_times_max.print_formatted(std::cout, 10, false, 16, "0.0000000000");
      std::cout << std::endl;

      std::cout << "AVG:" << std::endl;
      mg_times_avg.print_formatted(std::cout, 10, false, 16, "0.0000000000");
      std::cout << std::endl;

      std::cout << "MIN-INCL:" << std::endl;
      mg_times_incl_min.print_formatted(
        std::cout, 10, false, 16, "0.0000000000");
      std::cout << std::endl;

      std::cout << "MAX-INCL:" << std::endl;
      mg_times_incl_max.print_formatted(
        std::cout, 10, false, 16, "0.0000000000");
      std::cout << std::endl;

      std::cout << "AVG-INCL:" << std::endl;
      mg_times_incl_avg.print_formatted(
        std::cout, 10, false, 16, "0.0000000000");
      std::cout << std::endl;
    }

  table.add_value("time_pre", mg_times[0][1 + 0]);
  table.set_scientific("time_pre", true);
  table.add_value("time_residuum", mg_times[0][1 + 1]);
  table.set_scientific("time_residuum", true);
  table.add_value("time_res", mg_times[0][1 + 2]);
  table.set_scientific("time_res", true);
  table.add_value("time_cs", mg_times[0][1 + 3]);
  table.set_scientific("time_cs", true);
  table.add_value("time_pro", mg_times[0][1 + 4]);
  table.set_scientific("time_pro", true);
  table.add_value("time_edge_pro", mg_times[0][1 + 5]);
  table.set_scientific("time_edge_pro", true);
  table.add_value("time_post", mg_times[0][1 + 6]);
  table.set_scientific("time_post", true);

  table.add_value("time_to_mg",
                  mg_precon_timers[0].first / solver_control.last_step());
  table.set_scientific("time_to_mg", true);
  table.add_value("time_to_global",
                  mg_precon_timers[1].first / solver_control.last_step());
  table.set_scientific("time_to_global", true);

  monitor("mg_solve::5");
}



template <int dim,
          typename SystemMatrixType,
          typename LevelMatrixType,
          typename MGTransferType>
static void
mg_solve(SolverControl &                              solver_control,
         typename SystemMatrixType::VectorType &      dst,
         const typename SystemMatrixType::VectorType &src,
         const MultigridParameters &                  mg_data,
         const DoFHandler<dim> &                      dof,
         const SystemMatrixType &                     fine_matrix,
         const MGLevelObject<LevelMatrixType> &       mg_matrices,
         const MGTransferType &                       mg_transfer,
         const bool                                   verbose,
         const MPI_Comm &                             sub_comm,
         ConvergenceTable &                           table)
{
  mg_solve<dim, SystemMatrixType, LevelMatrixType, MGTransferType>(
    solver_control,
    dst,
    src,
    mg_data,
    dof,
    dof,
    fine_matrix,
    mg_matrices,
    mg_transfer,
    mg_transfer,
    0,
    verbose,
    sub_comm,
    table);
}


template <typename LEVEL_NUMBER_TYPE,
          int dim,
          int n_components,
          typename Number>
void
solve_with_global_coarsening(
  const std::string &                                           type,
  const std::vector<std::shared_ptr<const Triangulation<dim>>> &triangulations,
  const DoFHandler<dim> &                                       dof_handler_in,
  const MultigridParameters &                                   mg_data,
  const Operator<dim, n_components, Number> &                   op,
  typename Operator<dim, n_components, Number>::VectorType &    dst,
  const typename Operator<dim, n_components, Number>::VectorType &src,
  const bool                                                      verbose,
  ConvergenceTable &                                              table)
{
  const auto comm     = dof_handler_in.get_communicator();
  auto       sub_comm = comm;

  monitor("solve_with_global_coarsening::0");

  {
    unsigned int cell_counter = 0;

    for (const auto &cell : triangulations[0]->active_cell_iterators())
      if (cell->is_locally_owned())
        cell_counter++;

    const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

#if DEBUG
    const auto t = Utilities::MPI::gather(comm, cell_counter);

    if (rank == 0)
      {
        for (const auto tt : t)
          std::cout << tt << " ";
        std::cout << std::endl;
      }
#endif

    const int temp = cell_counter == 0 ? -1 : rank;

    const unsigned int max_rank = Utilities::MPI::max(temp, comm);

    table.add_value("sub_comm_size", max_rank + 1);

    if (max_rank != Utilities::MPI::n_mpi_processes(comm) - 1)
      {
        const bool color = rank <= max_rank;
        MPI_Comm_split(comm, color, rank, &sub_comm);

        if (color == false)
          {
            MPI_Comm_free(&sub_comm);
            sub_comm = MPI_COMM_NULL;
          }
      }
  }

  {
    monitor("solve_with_global_coarsening::1");

    const auto level_degrees =
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        dof_handler_in.get_fe().degree,
        MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
          bisect);

    const unsigned int min_level = 0;
    const unsigned int max_level = [&]() -> unsigned int {
      if (type == "PMG")
        return level_degrees.size() - 1;
      else if (type == "HMG-global")
        return triangulations.size() - 1;
      else if (type == "HPMG")
        return level_degrees.size() + triangulations.size() - 2;

      AssertThrow(false, ExcNotImplemented());

      return 0;
    }();

    using LevelOperatorType = Operator<dim, n_components, LEVEL_NUMBER_TYPE>;

    MGLevelObject<DoFHandler<dim>> dof_handlers(min_level, max_level);
    MGLevelObject<AffineConstraints<typename LevelOperatorType::value_type>>
      constraints(min_level, max_level);
    MGLevelObject<
      MGTwoLevelTransfer<dim, typename LevelOperatorType::VectorType>>
                                     transfers(min_level, max_level);
    MGLevelObject<LevelOperatorType> operators(min_level, max_level);

    MappingQ1<dim> mapping;

    monitor("solve_with_global_coarsening::2");

    for (auto l = min_level; l <= max_level; ++l)
      {
        auto &dof_handler = dof_handlers[l];
        auto &constraint  = constraints[l];
        auto &op          = operators[l];

        const auto degree = [&]() -> unsigned int {
          if (type == "PMG")
            return level_degrees[l];
          else if (type == "HMG-global")
            return level_degrees.back();
          else if (type == "HPMG")
            return level_degrees[std::max<int>(
              0, static_cast<int>(l) - triangulations.size() + 1)];

          AssertThrow(false, ExcNotImplemented());

          return 0;
        }();

        const FESystem<dim> fe(FE_Q<dim>{degree},
                               dof_handler_in.get_fe().n_components());
        const QGauss<dim>   quad(fe.degree + 1);

        const auto &tria = [&]() -> const Triangulation<dim> & {
          if (type == "PMG")
            return *triangulations.back();
          else if (type == "HMG-global")
            return *triangulations[l];
          else if (type == "HPMG")
            return *triangulations[std::min<unsigned int>(
              l, triangulations.size() - 1)];

          AssertThrow(false, ExcNotImplemented());

          return *triangulations.back();
        }();

        dof_handler.reinit(tria);
        dof_handler.distribute_dofs(fe);

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                locally_relevant_dofs);
        constraint.reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(
          mapping,
          dof_handler,
          0,
          Functions::ZeroFunction<dim, typename LevelOperatorType::value_type>(
            dof_handler_in.get_fe().n_components()),
          constraint);
        DoFTools::make_hanging_node_constraints(dof_handler, constraint);
        constraint.close();

        op.reinit(mapping, dof_handler, quad, constraint);
      }

    monitor("solve_with_global_coarsening::3");

    for (unsigned int l = min_level; l < max_level; ++l)
      transfers[l + 1].reinit(dof_handlers[l + 1],
                              dof_handlers[l],
                              constraints[l + 1],
                              constraints[l]);

    ConvergenceTable table_;
    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        table_.add_value(
          "cells", dof_handlers[l].get_triangulation().n_global_active_cells());
        table_.add_value("dofs", dof_handlers[l].n_dofs());
      }

    if (verbose && Utilities::MPI::this_mpi_process(
                     dof_handler_in.get_communicator()) == 0)
      table_.write_text(std::cout);

    MGTransferGlobalCoarsening<dim, typename LevelOperatorType::VectorType>
      transfer(transfers, [&](const auto l, auto &vec) {
        operators[l].initialize_dof_vector(vec);
      });

    monitor("solve_with_global_coarsening::4");

    ReductionControl solver_control(mg_data.do_parameter_study ?
                                      mg_data.cg_parameter_study.maxiter :
                                      mg_data.cg_normal.maxiter,
                                    mg_data.do_parameter_study ?
                                      mg_data.cg_parameter_study.abstol :
                                      mg_data.cg_normal.abstol,
                                    mg_data.do_parameter_study ?
                                      mg_data.cg_parameter_study.reltol :
                                      mg_data.cg_normal.reltol,
                                    false,
                                    false);

    MPI_Barrier(MPI_COMM_WORLD);

    mg_solve(solver_control,
             dst,
             src,
             mg_data,
             dof_handler_in,
             op,
             operators,
             transfer,
             verbose,
             sub_comm,
             table);

    monitor("solve_with_global_coarsening::5");
  }

  if (comm != sub_comm && sub_comm != MPI_COMM_NULL)
    MPI_Comm_free(&sub_comm);

  if (verbose)
    {
      const auto stats = MGTools::print_multigrid_statistics(triangulations);
      for (const auto stat : stats)
        {
          table.add_value(stat.first, stat.second);
          table.set_scientific(stat.first, true);
        }
    }
}



template <typename LEVEL_NUMBER_TYPE,
          int dim,
          int n_components,
          typename Number>
void
solve_with_local_smoothing(
  const std::string &                                       type,
  const DoFHandler<dim> &                                   dof_handler_in,
  const MultigridParameters &                               mg_data,
  const Operator<dim, n_components, Number> &               op,
  typename Operator<dim, n_components, Number>::VectorType &dst,
  const typename Operator<dim, n_components, Number>::VectorType &src,
  const bool                                                      verbose,
  ConvergenceTable &                                              table)
{
  using LevelOperatorType = Operator<dim, n_components, LEVEL_NUMBER_TYPE>;

  const bool do_pmg = type == "HPMG-local";

  const auto level_degrees =
    do_pmg ?
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        dof_handler_in.get_fe().degree,
        MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
          bisect) :
      std::vector<unsigned int>{dof_handler_in.get_fe().degree};

  std::vector<DoFHandler<dim>> dof_handlers(level_degrees.size());

  for (unsigned int i = 0; i < level_degrees.size(); ++i)
    {
      auto &dof_handler = dof_handlers[i];

      const FESystem<dim> fe(FE_Q<dim>{level_degrees[i]},
                             dof_handler_in.get_fe().n_components());

      dof_handler.reinit(dof_handler_in.get_triangulation());
      dof_handler.distribute_dofs(fe);

      if (i == 0)
        dof_handler.distribute_mg_dofs();
    }

  table.add_value("sub_comm_size",
                  Utilities::MPI::n_mpi_processes(
                    dof_handler_in.get_communicator()));

  const unsigned int min_level = 0;
  const unsigned int max_level =
    (dof_handler_in.get_triangulation().n_global_levels() - 1) +
    (do_pmg ? level_degrees.size() : 0);

  MGLevelObject<AffineConstraints<typename LevelOperatorType::value_type>>
                                   constraints(min_level, max_level);
  MGLevelObject<LevelOperatorType> operators(min_level, max_level);

  MGLevelObject<MGTwoLevelTransfer<dim, typename LevelOperatorType::VectorType>>
    transfers(std::min(max_level,
                       dof_handler_in.get_triangulation().n_global_levels()),
              max_level);

  MappingQ1<dim> mapping;

  MGConstrainedDoFs mg_constrained_dofs;

  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.initialize(dof_handlers.front());
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handlers.front(),
                                                     dirichlet_boundary);

  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners(
    dof_handler_in.get_triangulation().n_global_levels());

  monitor("solve_with_local_smoothing::0");

  for (auto l = min_level;
       l <= dof_handler_in.get_triangulation().n_global_levels() - 1;
       ++l)
    {
      auto &constraint = constraints[l];
      auto &op         = operators[l];

      QGauss<dim> quad(dof_handlers.front().get_fe().degree + 1);

      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handlers.front(),
                                                    l,
                                                    relevant_dofs);
      constraint.reinit(relevant_dofs);
      constraint.add_lines(mg_constrained_dofs.get_boundary_indices(l));
      constraint.close();

      op.reinit(mapping, dof_handlers.front(), quad, constraint, l);

      partitioners[l] = op.get_vector_partitioner();
    }

  if (do_pmg)
    for (auto l = dof_handler_in.get_triangulation().n_global_levels();
         l <= max_level;
         ++l)
      {
        auto &constraint = constraints[l];
        auto &op         = operators[l];
        auto &dof_handler =
          dof_handlers[l -
                       dof_handler_in.get_triangulation().n_global_levels()];

        QGauss<dim> quad(dof_handler.get_fe().degree + 1);

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                locally_relevant_dofs);
        constraint.reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(
          mapping,
          dof_handler,
          0,
          Functions::ZeroFunction<dim, typename LevelOperatorType::value_type>(
            dof_handler_in.get_fe().n_components()),
          constraint);
        DoFTools::make_hanging_node_constraints(dof_handler, constraint);
        constraint.close();

        op.reinit(mapping, dof_handler, quad, constraint);
      }

  monitor("solve_with_local_smoothing::1");

  MGTransferMatrixFree<dim, typename LevelOperatorType::value_type> transfer_ls(
    mg_constrained_dofs);
  transfer_ls.build(dof_handlers.front(), partitioners);

  if (do_pmg)
    for (auto l = dof_handler_in.get_triangulation().n_global_levels(), c = 0;
         l < max_level;
         ++l, ++c)
      transfers[l + 1].reinit(dof_handlers[c + 1],
                              dof_handlers[c],
                              constraints[l + 1],
                              constraints[l]);

  MGTransferGlobalCoarsening<dim, typename LevelOperatorType::VectorType>
    transfer_gc(transfers, [&](const auto l, auto &vec) {
      operators[l].initialize_dof_vector(vec);
    });

  monitor("solve_with_local_smoothing::2");

  ReductionControl solver_control(mg_data.do_parameter_study ?
                                    mg_data.cg_parameter_study.maxiter :
                                    mg_data.cg_normal.maxiter,
                                  mg_data.do_parameter_study ?
                                    mg_data.cg_parameter_study.abstol :
                                    mg_data.cg_normal.abstol,
                                  mg_data.do_parameter_study ?
                                    mg_data.cg_parameter_study.reltol :
                                    mg_data.cg_normal.reltol,
                                  false,
                                  false);

  if (do_pmg)
    mg_solve(solver_control,
             dst,
             src,
             mg_data,
             dof_handlers.back(),
             dof_handlers.front(),
             op,
             operators,
             transfer_gc,
             transfer_ls,
             dof_handler_in.get_triangulation().n_global_levels(),
             verbose,
             dof_handlers.front().get_communicator(),
             table);
  else
    mg_solve(solver_control,
             dst,
             src,
             mg_data,
             dof_handlers.front(),
             op,
             operators,
             transfer_ls,
             verbose,
             dof_handlers.front().get_communicator(),
             table);

  monitor("solve_with_local_smoothing::3");

  if (verbose)
    {
      const auto stats =
        MGTools::print_multigrid_statistics(dof_handler_in.get_triangulation());
      for (const auto stat : stats)
        {
          table.add_value(stat.first, stat.second);
          table.set_scientific(stat.first, true);
        }
    }
}



template <typename OperatorType, typename VectorType>
void
solve_with_amg(const std::string &        type,
               const MultigridParameters &mg_data,
               const OperatorType &       op,
               VectorType &               dst,
               const VectorType &         src,
               ConvergenceTable &         table)
{
  ReductionControl solver_control(mg_data.do_parameter_study ?
                                    mg_data.cg_parameter_study.maxiter :
                                    mg_data.cg_normal.maxiter,
                                  mg_data.do_parameter_study ?
                                    mg_data.cg_parameter_study.abstol :
                                    mg_data.cg_normal.abstol,
                                  mg_data.do_parameter_study ?
                                    mg_data.cg_parameter_study.reltol :
                                    mg_data.cg_normal.reltol,
                                  false,
                                  false);

  double time = 0.0;

  if (type == "AMG")
    {
      TrilinosWrappers::PreconditionAMG                 preconditioner;
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      preconditioner.initialize(op.get_trilinos_system_matrix(), data);

      {
        ScopedTimer timer(time);
        dealii::SolverCG<VectorType>(solver_control)
          .solve(op.get_trilinos_system_matrix(), dst, src, preconditioner);
      }
    }
#ifdef DEAL_II_WITH_PETSC
  else if (type == "AMGPETSc")
    {
      PETScWrappers::PreconditionBoomerAMG                 preconditioner;
      PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
      preconditioner.initialize(op.get_petsc_system_matrix(), data);

      {
        ScopedTimer timer(time);
        apply_petsc_operation(
          dst, src, MPI_COMM_WORLD, [&](auto &dst, const auto &src) {
            ScopedTimer timer(time);
            PETScWrappers::SolverCG(solver_control)
              .solve(op.get_petsc_system_matrix(), dst, src, preconditioner);
          });
      }
    }
#endif
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  table.add_value("sub_comm_size", 0);
  table.add_value("n_levels", 0);
  table.add_value("n_iterations", solver_control.last_step());
  table.add_value("time", time);
  table.add_value("time_cg", 0);
  table.add_value("throughput", src.size() * solver_control.last_step() / time);
  table.set_scientific("throughput", true);
  table.add_value("time_pre", 0);
  table.add_value("time_residuum", 0);
  table.add_value("time_res", 0);
  table.add_value("time_cs", 0);
  table.add_value("time_pro", 0);
  table.add_value("time_edge_pro", 0);
  table.add_value("time_post", 0);
  table.add_value("time_to_mg", 0);
  table.add_value("time_to_global", 0);

  table.add_value("workload_eff", 0);
  table.add_value("workload_path_max", 0);
  table.add_value("vertical_eff", 0);
  table.add_value("horizontal_eff", 0);
  table.add_value("mem_total", 0);
}



struct RunParameters
{
  std::string  type            = "PMG";
  std::string  geometry_type   = "quadrant_flexible";
  unsigned int n_ref_global    = 6;
  unsigned int n_ref_local     = 0;
  unsigned int fe_degree_fine  = 4;
  bool         paraview        = false;
  bool         verbose         = true;
  unsigned int p               = 0;
  std::string  policy_name     = "";
  std::string  mg_number_tyep  = "float";
  std::string  simulation_type = "Constant";

  int min_level   = -1;
  int min_n_cells = -1;

  MultigridParameters mg_data;

  void
  parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    prm.add_parameter("Type", type);
    prm.add_parameter("GeometryType", geometry_type);
    prm.add_parameter("NRefGlobal", n_ref_global);
    prm.add_parameter("NRefLocal", n_ref_local);
    prm.add_parameter("Degree", fe_degree_fine);
    prm.add_parameter("Paraview", paraview);
    prm.add_parameter("Verbosity", verbose);
    prm.add_parameter("Partitioner", p);
    prm.add_parameter("PartitionerName", policy_name);
    prm.add_parameter("MinLevel", min_level);
    prm.add_parameter("MinNCells", min_n_cells);
    prm.add_parameter("CoarseGridSolverType", mg_data.coarse_solver.type);
    prm.add_parameter("SmootherDegree", mg_data.smoother.degree);
    prm.add_parameter("CoarseSolverNCycles", mg_data.coarse_solver.n_cycles);
    prm.add_parameter("RelativeTolerance", mg_data.cg_normal.reltol);
    prm.add_parameter("MGNumberType", mg_number_tyep);
    prm.add_parameter("SimulationType", simulation_type);

    std::ifstream file;
    file.open(file_name);
    prm.parse_input_from_json(file, true);
  }
};



template <int dim,
          int n_components,
          typename Number = double,
          typename LEVEL_NUMBER_TYPE>
void
run(const RunParameters &params, ConvergenceTable &table)
{
  const std::string  type            = params.type;
  const std::string  geometry_type   = params.geometry_type;
  const unsigned int n_ref_global    = params.n_ref_global;
  const unsigned int n_ref_local     = params.n_ref_local;
  const unsigned int fe_degree_fine  = params.fe_degree_fine;
  const bool         paraview        = params.paraview;
  const bool         verbose         = params.verbose;
  const unsigned int p               = params.p;
  std::string        policy_name     = params.policy_name;
  const std::string  simulation_type = params.simulation_type;

  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  monitor("run::1");

  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD,
    Triangulation<dim>::MeshSmoothing::none,
    (type == "HMG-local" || type == "HPMG-local") ?
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy :
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

  if (geometry_type == "quadrant_flexible")
    GridGenerator::create_quadrant_(tria, n_ref_global, n_ref_local);
  else if (geometry_type == "quadrant")
    GridGenerator::create_quadrant(tria, n_ref_global);
  else if (geometry_type == "circle")
    GridGenerator::create_circle(tria, n_ref_global);
  else if (geometry_type == "annulus")
    GridGenerator::create_annulus(tria, n_ref_global);
  else if (geometry_type == "hypercube")
    {
      GridGenerator::hyper_cube(tria, -1.0, +1.0);
      tria.refine_global(n_ref_global);
    }
  else
    AssertThrow(false, ExcNotImplemented());

  monitor("run::2-" + std::to_string(tria.n_global_active_cells()));

  std::unique_ptr<RepartitioningPolicyTools::Base<dim>> policy;

  bool repartition_fine_triangulation = true;

  dealii::parallel::Helper<dim> helper(tria);

  if (type != "AMG" && type != "AMGPETSc")
    {
      if (policy_name == "")
        {
          switch (p)
            {
              case 0:
                policy_name = "DefaultPolicy";
                break;
              case 1:
                policy_name = "MinimalGranularityPolicy-40";
                break;
              case 2:
                policy_name = "CellWeightPolicy-1.0";
                break;
              case 3:
                policy_name = "CellWeightPolicy-1.5";
                break;
              case 4:
                policy_name = "CellWeightPolicy-2.0";
                break;
              case 5:
                policy_name = "CellWeightPolicy-2.5";
                break;
              case 6:
                policy_name = "FirstChildPolicy";
                break;
              case 7:
                policy_name = "BalancedGranularityPartitionPolicy";
                break;
              default:
                AssertThrow(false, ExcNotImplemented());
            }
        }

      const auto is_prefix = [](const std::string &label,
                                const std::string &prefix) -> bool {
        return label.rfind(prefix, 0) == 0;
      };

      const auto get_parameters =
        [](const std::string &str) -> std::vector<std::string> {
        std::stringstream        ss(str);
        std::vector<std::string> result;

        while (ss.good())
          {
            std::string substr;
            getline(ss, substr, '-');
            result.push_back(substr);
          }

        return result;
      };

      if (policy_name == "DefaultPolicy")
        {
          policy =
            std::make_unique<RepartitioningPolicyTools::DefaultPolicy<dim>>();
        }
      else if (policy_name == "BalancedGranularityPartitionPolicy")
        {
          policy = std::make_unique<
            RepartitioningPolicyTools::BalancedGranularityPartitionPolicy<dim>>(
            Utilities::MPI::n_mpi_processes(tria.get_communicator()));
          repartition_fine_triangulation = false;
        }
      else if (is_prefix(policy_name, "MinimalGranularityPolicy"))
        {
          policy = std::make_unique<
            RepartitioningPolicyTools::MinimalGranularityPolicy<dim>>(
            atoi(get_parameters(policy_name)[1].c_str()));
        }
      else if (is_prefix(policy_name, "CellWeightPolicy"))
        {
          const auto weight_function = parallel::hanging_nodes_weighting<dim>(
            helper, atof(get_parameters(policy_name)[1].c_str()));
          tria.signals.cell_weight.connect(weight_function);
          tria.repartition();

          policy =
            std::make_unique<RepartitioningPolicyTools::DefaultPolicy<dim>>(
              true);
        }
      else if (is_prefix(policy_name, "FirstChildPolicy"))
        {
          if (get_parameters(policy_name).size() > 1)
            {
              const auto weight_function =
                parallel::hanging_nodes_weighting<dim>(
                  helper, atof(get_parameters(policy_name)[1].c_str()));
              tria.signals.cell_weight.connect(weight_function);
              tria.repartition();
            }

          policy =
            std::make_unique<RepartitioningPolicyTools::FirstChildPolicy<dim>>(
              tria);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

  types::global_cell_index n_cells_w_hn  = 0;
  types::global_cell_index n_cells_wo_hn = 0;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        if (helper.is_constrained(cell))
          n_cells_w_hn++;
        else
          n_cells_wo_hn++;
      }

  n_cells_w_hn  = Utilities::MPI::sum(n_cells_w_hn, MPI_COMM_WORLD);
  n_cells_wo_hn = Utilities::MPI::sum(n_cells_wo_hn, MPI_COMM_WORLD);

  monitor("run::3");

  std::vector<std::shared_ptr<const Triangulation<dim>>> triangulations;

  if (type == "HMG-local" || type == "HPMG-local")
    {
      const auto partitions = policy->partition(tria);

      const auto description = TriangulationDescription::Utilities::
        create_description_from_triangulation(
          tria,
          partitions,
          TriangulationDescription::Settings::construct_multigrid_hierarchy);

      const auto new_triangulation =
        std::make_shared<parallel::fullydistributed::Triangulation<dim>>(
          tria.get_communicator());
      new_triangulation->create_triangulation(description);

      triangulations.emplace_back(new_triangulation);
    }
  else if (type == "AMG" || type == "AMGPETSc")
    {
      triangulations.emplace_back(&tria, [](auto *) {});
    }
  else
    {
      triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          tria,
          *policy,
          false /*=preserve_fine_triangulation*/,
          repartition_fine_triangulation);
    }

  if (triangulations.size() > 1)
    {
      // collect levels
      std::vector<std::shared_ptr<const Triangulation<dim>>> temp;

      // find first relevant coarse-grid triangulation
      auto ptr = std::find_if(
        triangulations.begin(),
        triangulations.end() - 1,
        [&params](const auto &tria) {
          if (params.min_level != -1) // minimum number of levels
            {
              if (params.min_level <= static_cast<int>(tria->n_global_levels()))
                return true;
            }
          else if (params.min_n_cells != -1) // minimum number of cells
            {
              if (static_cast<int>(tria->n_global_active_cells()) >=
                  params.min_n_cells)
                return true;
            }
          else
            {
              return true;
            }
          return false;
        });

      // consider all triangulations from that one
      while (ptr != triangulations.end())
        temp.push_back(*(ptr++));

      triangulations = temp;
    }

  DoFHandler<dim> dof_handler(*triangulations.back());

  AffineConstraints<Number>           constraint;
  Operator<dim, n_components, Number> op;

  MappingQ1<dim> mapping;

  const FESystem<dim> fe(FE_Q<dim>{fe_degree_fine}, n_components);
  const QGauss<dim>   quad(fe_degree_fine + 1);

  monitor("run::4");

  dof_handler.distribute_dofs(fe);

  monitor("run::5-" + std::to_string(dof_handler.n_dofs()));

  if (type == "HMG-local" || type == "HPMG-local")
    dof_handler.distribute_mg_dofs();

  monitor("run::6");

  std::shared_ptr<Function<dim, Number>> dbc_func;
  std::shared_ptr<Function<dim, Number>> rhs_func;

  if (simulation_type == "Constant")
    {
      rhs_func = std::make_shared<Functions::ConstantFunction<dim, Number>>(
        1.0, n_components);
      dbc_func = std::make_shared<Functions::ZeroFunction<dim, Number>>(1.0);
    }
  else if (simulation_type == "Gaussian")
    {
      const std::vector<Point<dim>> points = {Point<dim>(-0.5, -0.5, -0.5)};
      const double                  width  = 0.1;

      rhs_func = std::make_shared<GaussianRightHandSide<dim>>(points, width);
      dbc_func = std::make_shared<GaussianSolution<dim>>(points, width);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  constraint.reinit(locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, *dbc_func, constraint);
  DoFTools::make_hanging_node_constraints(dof_handler, constraint);
  constraint.close();

  monitor("run::7");

  op.reinit(mapping, dof_handler, quad, constraint);

  const MultigridParameters &mg_data = params.mg_data; // TODO

  VectorType solution, rhs;
  op.initialize_dof_vector(solution);
  op.initialize_dof_vector(rhs);

  op.rhs(rhs, rhs_func, mapping, dof_handler, quad);

  monitor("run::8");

  table.add_value("dim", dim);
  table.add_value("n_cells", triangulations.back()->n_global_active_cells());
  table.add_value("n_cells_hn", n_cells_w_hn);
  table.add_value("n_cells_n", n_cells_wo_hn);
  table.add_value("degree", fe_degree_fine);
  table.add_value("n_ref_global", n_ref_global);
  table.add_value("n_ref_local", n_ref_local);
  table.add_value("n_dofs", dof_handler.n_dofs());

  if (type == "AMG" || type == "AMGPETSc")
    solve_with_amg(type, mg_data, op, solution, rhs, table);
  else if (type == "PMG" || type == "HMG-global" || type == "HPMG")
    solve_with_global_coarsening<LEVEL_NUMBER_TYPE>(type,
                                                    triangulations,
                                                    dof_handler,
                                                    mg_data,
                                                    op,
                                                    solution,
                                                    rhs,
                                                    verbose,
                                                    table);
  else if (type == "HMG-local" || type == "HPMG-local")
    solve_with_local_smoothing<LEVEL_NUMBER_TYPE>(
      type, dof_handler, mg_data, op, solution, rhs, verbose, table);
  else
    AssertThrow(false, ExcNotImplemented());

  monitor("run::9");

  monitor("break");

  if (paraview == false)
    return;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  DataOut<dim> data_out;
  data_out.set_flags(flags);

  data_out.attach_dof_handler(dof_handler);


  solution.update_ghost_values();
  constraint.distribute(solution);
  data_out.add_data_vector(solution, "solution");


  data_out.build_patches(mapping, 3);

  data_out.write_vtu_in_parallel("multigrid.vtu", MPI_COMM_WORLD);
}

int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

      const MPI_Comm comm = MPI_COMM_WORLD;

      dealii::ConditionalOStream pcout(
        std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

      if (argc == 1)
        {
          if (pcout.is_active())
            printf("ERROR: No .json parameter files has been provided!\n");

          return 1;
        }

      dealii::deallog.depth_console(0);

      ConvergenceTable table;

      for (int i = 1; i < argc; i++)
        {
          if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << std::string(argv[i]) << std::endl;

          RunParameters params;
          params.parse(std::string(argv[i]));

          if (params.mg_number_tyep == "double")
            run<3, 1, double, double>(params, table);
          else if (params.mg_number_tyep == "float")
            run<3, 1, double, float>(params, table);
          else
            AssertThrow(false, ExcNotImplemented());

          if (pcout.is_active())
            table.write_text(pcout.get_stream());
        }

      if (pcout.is_active())
        table.write_text(pcout.get_stream());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
