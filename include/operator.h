#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

using namespace dealii;


template <int dim_, int n_components, typename Number>
class LaplaceOperator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using size_type  = types::global_dof_index;

  static const int dim = dim_;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, number>;

  void
  reinit(const Mapping<dim> &             mapping,
         const DoFHandler<dim> &          dof_handler,
         const Quadrature<dim> &          quad,
         const AffineConstraints<number> &constraints,
         const unsigned int mg_level = numbers::invalid_unsigned_int)
  {
#ifdef DEAL_II_WITH_TRILINOS
    this->trilinos_system_matrix.clear();
#endif

    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_quadrature_points | update_gradients |
                                update_values | update_normal_vectors;
    data.mg_level = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);

    constrained_indices.clear();
    for (auto i : this->matrix_free.get_constrained_dofs())
      constrained_indices.push_back(i);
    constrained_values.resize(constrained_indices.size());

    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      {
        std::vector<types::global_dof_index> interface_indices;
        IndexSet                             refinement_edge_indices;
        refinement_edge_indices = get_refinement_edges(this->matrix_free);
        refinement_edge_indices.fill_index_vector(interface_indices);

        edge_constrained_indices.clear();
        edge_constrained_indices.reserve(interface_indices.size());
        edge_constrained_values.resize(interface_indices.size());
        const IndexSet &locally_owned =
          this->matrix_free.get_dof_handler().locally_owned_mg_dofs(
            this->matrix_free.get_mg_level());
        for (unsigned int i = 0; i < interface_indices.size(); ++i)
          if (locally_owned.is_element(interface_indices[i]))
            edge_constrained_indices.push_back(
              locally_owned.index_within_set(interface_indices[i]));

        this->has_edge_constrained_indices =
          Utilities::MPI::max(edge_constrained_indices.size(),
                              dof_handler.get_communicator()) > 0;

        if (this->has_edge_constrained_indices)
          {
            edge_constrained_cell.resize(matrix_free.n_cell_batches(), false);

            VectorType temp;
            matrix_free.initialize_dof_vector(temp);

            for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
              temp.local_element(edge_constrained_indices[i]) = 1.0;

            temp.update_ghost_values();

            FECellIntegrator integrator(matrix_free);

            for (unsigned int cell = 0; cell < matrix_free.n_cell_batches();
                 ++cell)
              {
                integrator.reinit(cell);
                integrator.read_dof_values(temp);

                for (unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
                  if ((integrator.begin_dof_values()[i] ==
                       VectorizedArray<Number>()) == false)
                    {
                      edge_constrained_cell[cell] = true;
                      break;
                    }
              }


#ifdef DEBUG
            unsigned int count = 0;
            for (const auto i : edge_constrained_cell)
              if (i)
                count++;

            const unsigned int count_global =
              Utilities::MPI::sum(count, dof_handler.get_communicator());

            const unsigned int count_cells_global =
              Utilities::MPI::sum(matrix_free.n_cell_batches(),
                                  dof_handler.get_communicator());

            if (Utilities::MPI::this_mpi_process(
                  dof_handler.get_communicator()) == 0)
              std::cout << count_global << " " << count_cells_global
                        << std::endl;
#endif
          }
      }
  }

  virtual types::global_dof_index
  m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(
        this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return 0;
  }

  virtual void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
    // save values for edge constrained dofs and set them to 0 in src vector
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        edge_constrained_values[i] = std::pair<number, number>(
          src.local_element(edge_constrained_indices[i]),
          dst.local_element(edge_constrained_indices[i]));

        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(edge_constrained_indices[i]) = 0.;
      }

    this->matrix_free.cell_loop(
      &LaplaceOperator::do_cell_integral_range, this, dst, src, true);

    // set constrained dofs as the sum of current dst value and src value
    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);

    // restoring edge constrained dofs in src and dst
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
        dst.local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
      }
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  vmult_interface_down(VectorType &dst, VectorType const &src) const
  {
    this->matrix_free.cell_loop(
      &LaplaceOperator::do_cell_integral_range, this, dst, src, true);

    // set constrained dofs as the sum of current dst value and src value
    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }

  void
  vmult_interface_up(VectorType &dst, VectorType const &src) const
  {
    if (has_edge_constrained_indices == false)
      {
        dst = Number(0.);
        return;
      }

    dst = 0.0;

    // make a copy of src vector and set everything to 0 except edge
    // constrained dofs
    VectorType src_cpy;
    src_cpy.reinit(src, /*omit_zeroing_entries=*/false);

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      src_cpy.local_element(edge_constrained_indices[i]) =
        src.local_element(edge_constrained_indices[i]);

    // do loop with copy of src
    this->matrix_free.cell_loop(&LaplaceOperator::do_cell_integral_range<true>,
                                this,
                                dst,
                                src_cpy,
                                false);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &LaplaceOperator::do_cell_integral_local,
                                      this);

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      diagonal.local_element(edge_constrained_indices[i]) = 0.0;

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual const TrilinosWrappers::SparseMatrix &
  get_trilinos_system_matrix() const
  {
    return get_trilinos_system_matrix(matrix_free.get_task_info().communicator);
  }

  virtual const TrilinosWrappers::SparseMatrix &
  get_trilinos_system_matrix(const MPI_Comm comm) const
  {
    if (comm != MPI_COMM_NULL && trilinos_system_matrix.m() == 0 &&
        trilinos_system_matrix.n() == 0)
      {
        // Set up sparsity pattern of system matrix.
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        TrilinosWrappers::SparsityPattern dsp(
          this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(
              this->matrix_free.get_mg_level()) :
            dof_handler.locally_owned_dofs(),
          comm);

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         this->constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);

        dsp.compress();
        trilinos_system_matrix.reinit(dsp);

        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          trilinos_system_matrix,
          &LaplaceOperator::do_cell_integral_local,
          this);
      }

    return this->trilinos_system_matrix;
  }
#endif


#ifdef DEAL_II_WITH_PETSC
  virtual const PETScWrappers::MPI::SparseMatrix &
  get_petsc_system_matrix() const
  {
    return get_petsc_system_matrix(matrix_free.get_task_info().communicator);
  }



  virtual const PETScWrappers::MPI::SparseMatrix &
  get_petsc_system_matrix(const MPI_Comm mpi_communicator) const
  {
    // Check if matrix has already been set up.
    if (petsc_system_matrix.m() == 0 && petsc_system_matrix.n() == 0)
      {
        // Set up sparsity pattern of system matrix.
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        const auto locally_owned_dofs =
          this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(
              this->matrix_free.get_mg_level()) :
            dof_handler.locally_owned_dofs();

        IndexSet locally_relevant_dofs;

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          {
            DoFTools::extract_locally_relevant_level_dofs(
              dof_handler,
              this->matrix_free.get_mg_level(),
              locally_relevant_dofs);
          }
        else
          {
            DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);
          }

        DynamicSparsityPattern dsp(locally_relevant_dofs);

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         this->constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);

        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   locally_owned_dofs,
                                                   mpi_communicator,
                                                   locally_relevant_dofs);
        petsc_system_matrix.reinit(locally_owned_dofs,
                                   locally_owned_dofs,
                                   dsp,
                                   mpi_communicator);

        // Assemble system matrix.
        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          petsc_system_matrix,
          &LaplaceOperator::do_cell_integral_local,
          this);
      }

    return this->petsc_system_matrix;
  }
#endif



  void
  rhs(VectorType &                          system_rhs,
      const std::shared_ptr<Function<dim>> &rhs_func,
      const Mapping<dim> &                  mapping,
      const DoFHandler<dim> &               dof_handler,
      const Quadrature<dim> &               quad) const
  {
    const int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [&rhs_func](
        const auto &matrix_free, auto &dst, const auto &, const auto cells) {
        FECellIntegrator phi(matrix_free, cells);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              if constexpr (n_components == 1)
                {
                  VectorizedArray<number> coeff = 0;

                  const auto point_batch = phi.quadrature_point(q);

                  for (unsigned int v = 0; v < VectorizedArray<number>::size();
                       ++v)
                    {
                      Point<dim> single_point;
                      for (unsigned int d = 0; d < dim; d++)
                        single_point[d] = point_batch[d][v];
                      coeff[v] = rhs_func->value(single_point);
                    }

                  phi.submit_value(coeff, q);
                }
              else
                {
                  Tensor<1, n_components> temp;

                  Assert(false, ExcNotImplemented());

                  for (int i = 0; i < n_components; ++i)
                    temp[i] = 1.0;

                  phi.submit_value(temp, q);
                }

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      system_rhs,
      dummy,
      true);

    AffineConstraints<number> constraints_without_dbc;
    constraints_without_dbc.reinit(
      DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints_without_dbc);
    constraints_without_dbc.close();

    VectorType b, x;

    this->initialize_dof_vector(b);
    this->initialize_dof_vector(x);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      update_values | update_gradients | update_quadrature_points;

    MatrixFree<dim, number> matrix_free;
    matrix_free.reinit(
      mapping, dof_handler, constraints_without_dbc, quad, data);

    // set constrained
    constraints.distribute(x);

    // perform matrix-vector multiplication (with unconstrained system and
    // constrained set in vector)
    matrix_free.cell_loop(
      &LaplaceOperator::do_cell_integral_range, this, b, x, true);

    // clear constrained values
    constraints.set_zero(b);

    // move to the right-hand side
    system_rhs -= b;
  }

private:
  void
  do_cell_integral_local(FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }

  void
  do_cell_integral_global(FECellIntegrator &integrator,
                          VectorType &      dst,
                          const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }

  template <bool apply_edge_optimization = false>
  void
  do_cell_integral_range(
    const MatrixFree<dim, number> &              matrix_free,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        if (apply_edge_optimization && (edge_constrained_cell[cell] == false))
          continue;

        integrator.reinit(cell);

        do_cell_integral_global(integrator, dst, src);
      }
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

#ifdef DEAL_II_WITH_TRILINOS
  mutable TrilinosWrappers::SparseMatrix trilinos_system_matrix;
#endif

#ifdef DEAL_II_WITH_PETSC
  mutable PETScWrappers::MPI::SparseMatrix petsc_system_matrix;
#endif

  /**
   * Constrained indices.
   *
   * @note Needed in matrix-free vmults.
   */
  std::vector<unsigned int> constrained_indices;

  /**
   * Constrained values.
   *
   * @note Needed in matrix-free vmults.
   */
  mutable std::vector<std::pair<number, number>> constrained_values;

  /**
   * Edge-constrained indices.
   *
   * @note Needed in matrix-free vmults.
   */
  std::vector<unsigned int> edge_constrained_indices;

  bool has_edge_constrained_indices = false;

  /**
   * Edge-constrained values.
   *
   * @note Needed in matrix-free vmults.
   */
  mutable std::vector<std::pair<number, number>> edge_constrained_values;

  std::vector<bool> edge_constrained_cell;

  static IndexSet
  get_refinement_edges(const MatrixFree<dim, number> &matrix_free)
  {
    const unsigned int level = matrix_free.get_mg_level();

    std::vector<IndexSet> refinement_edge_indices;
    refinement_edge_indices.clear();
    const unsigned int nlevels =
      matrix_free.get_dof_handler().get_triangulation().n_global_levels();
    refinement_edge_indices.resize(nlevels);
    for (unsigned int l = 0; l < nlevels; l++)
      refinement_edge_indices[l] =
        IndexSet(matrix_free.get_dof_handler().n_dofs(l));

    MGTools::extract_inner_interface_dofs(matrix_free.get_dof_handler(),
                                          refinement_edge_indices);
    return refinement_edge_indices[level];
  }
};


/**
 * The following operator implements the action of a classic linear elasticity
 * operator with:
 * - Homogeneous boundary conditions on boundary_id == 0
 * - Dirichlet boundary conditions on boundary_id == 1
 * - Inhomogeneous Neumann boundary conditions on boundary_id == 2
 *
 */
template <int dim_, int n_components, typename Number>
class ElasticityOperator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using size_type  = types::global_dof_index;

  static const int dim = dim_;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, number>;
  using FEFaceIntegrator = FEFaceEvaluation<dim, -1, 0, n_components, number>;
  // Set Lame moduli here
  ElasticityOperator(const double lambda_ = 1., const double mu_ = 1.)
  {
    Assert(dim == 3, ExcMessage("The elasticity example is 3D only."));
    Assert(mu > 0., ExcMessage("Shear modulus must be positive."));
    lambda = lambda_;
    mu     = mu_;
  }

  void
  reinit(const Mapping<dim> &             mapping,
         const DoFHandler<dim> &          dof_handler,
         const Quadrature<dim> &          quad,
         const AffineConstraints<number> &constraints,
         const unsigned int mg_level = numbers::invalid_unsigned_int)
  {
    Assert(
      dof_handler.get_triangulation().get_boundary_ids().size() == 3,
      ExcMessage(
        "The following example is configured with mixed boundary conditions. Check the original triangulation."));
#ifdef DEAL_II_WITH_TRILINOS
    this->trilinos_system_matrix.clear();
#endif

    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      update_quadrature_points | update_gradients | update_values;
    data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    data.mg_level = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);

    constrained_indices.clear();
    for (auto i : this->matrix_free.get_constrained_dofs())
      constrained_indices.push_back(i);
    constrained_values.resize(constrained_indices.size());

    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      {
        std::vector<types::global_dof_index> interface_indices;
        IndexSet                             refinement_edge_indices;
        refinement_edge_indices = get_refinement_edges(this->matrix_free);
        refinement_edge_indices.fill_index_vector(interface_indices);

        edge_constrained_indices.clear();
        edge_constrained_indices.reserve(interface_indices.size());
        edge_constrained_values.resize(interface_indices.size());
        const IndexSet &locally_owned =
          this->matrix_free.get_dof_handler().locally_owned_mg_dofs(
            this->matrix_free.get_mg_level());
        for (unsigned int i = 0; i < interface_indices.size(); ++i)
          if (locally_owned.is_element(interface_indices[i]))
            edge_constrained_indices.push_back(
              locally_owned.index_within_set(interface_indices[i]));

        this->has_edge_constrained_indices =
          Utilities::MPI::max(edge_constrained_indices.size(),
                              dof_handler.get_communicator()) > 0;

        if (this->has_edge_constrained_indices)
          {
            edge_constrained_cell.resize(matrix_free.n_cell_batches(), false);

            VectorType temp;
            matrix_free.initialize_dof_vector(temp);

            for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
              temp.local_element(edge_constrained_indices[i]) = 1.0;

            temp.update_ghost_values();

            FECellIntegrator integrator(matrix_free);

            for (unsigned int cell = 0; cell < matrix_free.n_cell_batches();
                 ++cell)
              {
                integrator.reinit(cell);
                integrator.read_dof_values(temp);

                for (unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
                  if ((integrator.begin_dof_values()[i] ==
                       VectorizedArray<Number>()) == false)
                    {
                      edge_constrained_cell[cell] = true;
                      break;
                    }
              }


#ifdef DEBUG
            unsigned int count = 0;
            for (const auto i : edge_constrained_cell)
              if (i)
                count++;

            const unsigned int count_global =
              Utilities::MPI::sum(count, dof_handler.get_communicator());

            const unsigned int count_cells_global =
              Utilities::MPI::sum(matrix_free.n_cell_batches(),
                                  dof_handler.get_communicator());

            if (Utilities::MPI::this_mpi_process(
                  dof_handler.get_communicator()) == 0)
              std::cout << count_global << " " << count_cells_global
                        << std::endl;
#endif
          }
      }
  }

  virtual types::global_dof_index
  m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(
        this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return 0;
  }

  virtual void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
    // save values for edge constrained dofs and set them to 0 in src vector
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        edge_constrained_values[i] = std::pair<number, number>(
          src.local_element(edge_constrained_indices[i]),
          dst.local_element(edge_constrained_indices[i]));

        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(edge_constrained_indices[i]) = 0.;
      }

    this->matrix_free.cell_loop(
      &ElasticityOperator::do_cell_integral_range, this, dst, src, true);

    // set constrained dofs as the sum of current dst value and src value
    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);

    // restoring edge constrained dofs in src and dst
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        const_cast<LinearAlgebra::distributed::Vector<number> &>(src)
          .local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
        dst.local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
      }
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  vmult_interface_down(VectorType &dst, VectorType const &src) const
  {
    this->matrix_free.cell_loop(
      &ElasticityOperator::do_cell_integral_range, this, dst, src, true);

    // set constrained dofs as the sum of current dst value and src value
    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }

  void
  vmult_interface_up(VectorType &dst, VectorType const &src) const
  {
    if (has_edge_constrained_indices == false)
      {
        dst = Number(0.);
        return;
      }

    dst = 0.0;

    // make a copy of src vector and set everything to 0 except edge
    // constrained dofs
    VectorType src_cpy;
    src_cpy.reinit(src, /*omit_zeroing_entries=*/false);

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      src_cpy.local_element(edge_constrained_indices[i]) =
        src.local_element(edge_constrained_indices[i]);

    // do loop with copy of src
    this->matrix_free.cell_loop(
      &ElasticityOperator::do_cell_integral_range<true>,
      this,
      dst,
      src_cpy,
      false);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(
      matrix_free, diagonal, &ElasticityOperator::do_cell_integral_local, this);

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      diagonal.local_element(edge_constrained_indices[i]) = 0.0;

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual const TrilinosWrappers::SparseMatrix &
  get_trilinos_system_matrix() const
  {
    return get_trilinos_system_matrix(matrix_free.get_task_info().communicator);
  }

  virtual const TrilinosWrappers::SparseMatrix &
  get_trilinos_system_matrix(const MPI_Comm comm) const
  {
    if (comm != MPI_COMM_NULL && trilinos_system_matrix.m() == 0 &&
        trilinos_system_matrix.n() == 0)
      {
        // Set up sparsity pattern of system matrix.
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        TrilinosWrappers::SparsityPattern dsp(
          this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(
              this->matrix_free.get_mg_level()) :
            dof_handler.locally_owned_dofs(),
          comm);

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         this->constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);

        dsp.compress();
        trilinos_system_matrix.reinit(dsp);

        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          trilinos_system_matrix,
          &ElasticityOperator::do_cell_integral_local,
          this);
      }

    return this->trilinos_system_matrix;
  }
#endif


#ifdef DEAL_II_WITH_PETSC
  virtual const PETScWrappers::MPI::SparseMatrix &
  get_petsc_system_matrix() const
  {
    return get_petsc_system_matrix(matrix_free.get_task_info().communicator);
  }



  virtual const PETScWrappers::MPI::SparseMatrix &
  get_petsc_system_matrix(const MPI_Comm mpi_communicator) const
  {
    // Check if matrix has already been set up.
    if (petsc_system_matrix.m() == 0 && petsc_system_matrix.n() == 0)
      {
        // Set up sparsity pattern of system matrix.
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        const auto locally_owned_dofs =
          this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(
              this->matrix_free.get_mg_level()) :
            dof_handler.locally_owned_dofs();

        IndexSet locally_relevant_dofs;

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          {
            DoFTools::extract_locally_relevant_level_dofs(
              dof_handler,
              this->matrix_free.get_mg_level(),
              locally_relevant_dofs);
          }
        else
          {
            DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);
          }

        DynamicSparsityPattern dsp(locally_relevant_dofs);

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         this->constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);

        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   locally_owned_dofs,
                                                   mpi_communicator,
                                                   locally_relevant_dofs);
        petsc_system_matrix.reinit(locally_owned_dofs,
                                   locally_owned_dofs,
                                   dsp,
                                   mpi_communicator);

        // Assemble system matrix.
        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          petsc_system_matrix,
          &ElasticityOperator::do_cell_integral_local,
          this);
      }

    return this->petsc_system_matrix;
  }
#endif



  void
  rhs(VectorType &                          system_rhs,
      const std::shared_ptr<Function<dim>> &rhs_func,
      const Mapping<dim> &                  mapping,
      const DoFHandler<dim> &               dof_handler,
      const Quadrature<dim> &               quad) const
  {
    const int    dummy    = 0;
    const double pressure = -1e5;
    (void)rhs_func;

    matrix_free.template loop<VectorType, int>(
      [](const auto &matrix_free, auto &dst, const auto &, const auto cells) {
        (void)matrix_free;
        (void)dst;
        (void)cells;
      },
      [](const auto &matrix_free, auto &dst, const auto &, const auto cells) {
        (void)matrix_free;
        (void)dst;
        (void)cells;
      },
      [&pressure](const auto &matrix_free,
                  auto &      dst,
                  const auto &,
                  const auto face_range) {
        FEFaceIntegrator phi(matrix_free, true);
        for (unsigned int face = face_range.first; face < face_range.second;
             ++face)
          {
            // pressure is applied on boundary id number 2
            if (matrix_free.get_boundary_id(face) == 2)
              {
                phi.reinit(face);
                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    phi.submit_value((pressure)*phi.normal_vector(q), q);
                  }
                phi.integrate_scatter(EvaluationFlags::values, dst);
              }
          }
      },
      system_rhs,
      dummy,
      true);

    AffineConstraints<number> constraints_without_dbc;
    constraints_without_dbc.reinit(
      DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints_without_dbc);
    constraints_without_dbc.close();

    VectorType b, x;

    this->initialize_dof_vector(b);
    this->initialize_dof_vector(x);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      update_values | update_gradients | update_quadrature_points;

    MatrixFree<dim, number> matrix_free;
    matrix_free.reinit(
      mapping, dof_handler, constraints_without_dbc, quad, data);

    // set constrained
    constraints.distribute(x);

    // perform matrix-vector multiplication (with unconstrained system and
    // constrained set in vector)
    matrix_free.cell_loop(
      &ElasticityOperator::do_cell_integral_range, this, b, x, true);

    // clear constrained values
    constraints.set_zero(b);

    // move to the right-hand side
    system_rhs -= b;
  }

private:
  void
  do_cell_integral_local(FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        const SymmetricTensor<2, dim, VectorizedArray<number>> sym_grad_u =
          integrator.get_symmetric_gradient(q);
        const auto lambda_div_u = lambda * trace(sym_grad_u);
        auto       stress       = 2. * mu * sym_grad_u;
        // Add lambda div(u)I
        for (unsigned int i = 0; i < dim; ++i)
          stress[i][i] += lambda_div_u;

        integrator.submit_symmetric_gradient(stress, q);
      }

    integrator.integrate(EvaluationFlags::gradients);
  }

  void
  do_cell_integral_global(FECellIntegrator &integrator,
                          VectorType &      dst,
                          const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        const SymmetricTensor<2, dim, VectorizedArray<number>> sym_grad_u =
          integrator.get_symmetric_gradient(q);
        const auto lambda_div_u = lambda * trace(sym_grad_u);
        auto       stress       = 2. * mu * sym_grad_u;
        // Add lambda div(u)I
        for (unsigned int i = 0; i < dim; ++i)
          stress[i][i] += lambda_div_u;

        integrator.submit_symmetric_gradient(stress, q);
      }
    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }

  template <bool apply_edge_optimization = false>
  void
  do_cell_integral_range(
    const MatrixFree<dim, number> &              matrix_free,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        if (apply_edge_optimization && (edge_constrained_cell[cell] == false))
          continue;

        integrator.reinit(cell);

        do_cell_integral_global(integrator, dst, src);
      }
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

#ifdef DEAL_II_WITH_TRILINOS
  mutable TrilinosWrappers::SparseMatrix trilinos_system_matrix;
#endif

#ifdef DEAL_II_WITH_PETSC
  mutable PETScWrappers::MPI::SparseMatrix petsc_system_matrix;
#endif

  /**
   * Constrained indices.
   *
   * @note Needed in matrix-free vmults.
   */
  std::vector<unsigned int> constrained_indices;

  /**
   * Constrained values.
   *
   * @note Needed in matrix-free vmults.
   */
  mutable std::vector<std::pair<number, number>> constrained_values;

  /**
   * Edge-constrained indices.
   *
   * @note Needed in matrix-free vmults.
   */
  std::vector<unsigned int> edge_constrained_indices;

  bool has_edge_constrained_indices = false;

  /**
   * Edge-constrained values.
   *
   * @note Needed in matrix-free vmults.
   */
  mutable std::vector<std::pair<number, number>> edge_constrained_values;

  std::vector<bool> edge_constrained_cell;

  /**
   * First Lame' module.
   */

  double lambda = 1.;

  /**
   * Shear modulus.
   */
  double mu = 1.;

  static IndexSet
  get_refinement_edges(const MatrixFree<dim, number> &matrix_free)
  {
    const unsigned int level = matrix_free.get_mg_level();

    std::vector<IndexSet> refinement_edge_indices;
    refinement_edge_indices.clear();
    const unsigned int nlevels =
      matrix_free.get_dof_handler().get_triangulation().n_global_levels();
    refinement_edge_indices.resize(nlevels);
    for (unsigned int l = 0; l < nlevels; l++)
      refinement_edge_indices[l] =
        IndexSet(matrix_free.get_dof_handler().n_dofs(l));

    MGTools::extract_inner_interface_dofs(matrix_free.get_dof_handler(),
                                          refinement_edge_indices);
    return refinement_edge_indices[level];
  }
};
