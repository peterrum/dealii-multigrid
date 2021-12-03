
namespace dealii::GridGenerator
{
  template <int dim>
  void
  create_point(Triangulation<dim> &tria, const unsigned int n_refinements)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0);

    for (int i = 0; i < std::min(static_cast<int>(n_refinements), 3); i++)
      tria.refine_global(1);

    for (unsigned int i = 3; i < n_refinements; i++)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
                   i++)
                if (cell->vertex(i).norm() < 1e-8)
                  cell->set_refine_flag();
            }
        tria.execute_coarsening_and_refinement();
      }

    AssertDimension(tria.n_global_levels() - 1, n_refinements);
  }



  template <int dim>
  void
  create_circle(Triangulation<dim> &tria, const unsigned int n_refinements)
  {
    // according to the description in A FLEXIBLE, PARALLEL, ADAPTIVE
    // GEOMETRIC MULTIGRID METHOD FOR FEM (Clevenger, Heister, Kanschat,
    // Kronbichler): https://arxiv.org/pdf/1904.03317.pdf

    hyper_cube(tria, -1.0, +1.0);

    for (int i = 0; i < std::min(static_cast<int>(n_refinements), 3); i++)
      tria.refine_global(1);

    for (unsigned int i = 3; i < n_refinements; i++)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
                   i++)
                if (cell->vertex(i).norm() < 1.0 / (4.0 * numbers::PI))
                  cell->set_refine_flag();
            }
        tria.execute_coarsening_and_refinement();
      }

    AssertDimension(tria.n_global_levels() - 1, n_refinements);
  }



  template <int dim>
  void
  create_quadrant(Triangulation<dim> &tria, const unsigned int n_refinements)
  {
    // according to the description in A FLEXIBLE, PARALLEL, ADAPTIVE
    // GEOMETRIC MULTIGRID METHOD FOR FEM (Clevenger, Heister, Kanschat,
    // Kronbichler): https://arxiv.org/pdf/1904.03317.pdf

    hyper_cube(tria, -1.0, +1.0);

    if (n_refinements == 0)
      return;

    tria.refine_global(1);

    for (unsigned int i = 1; i < n_refinements; i++)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              bool flag = true;
              for (int d = 0; d < dim; d++)
                if (cell->center()[d] > 0.0)
                  flag = false;
              if (flag)
                cell->set_refine_flag();
            }
        tria.execute_coarsening_and_refinement();
      }

    AssertDimension(tria.n_global_levels() - 1, n_refinements);
  }



  template <int dim>
  void
  create_quadrant_(Triangulation<dim> &tria,
                   const unsigned int  n_ref_global,
                   const unsigned int  n_ref_local)
  {
    GridGenerator::hyper_cube(tria, -1.0, +1.0);
    tria.refine_global(n_ref_global);

    for (unsigned int i = 0; i < n_ref_local; ++i)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              bool flag = true;
              for (int d = 0; d < dim; d++)
                if (cell->center()[d] > 0.0)
                  flag = false;
              if (flag)
                cell->set_refine_flag();
            }
        tria.execute_coarsening_and_refinement();
      }
  }



  template <int dim>
  void
  create_annulus(Triangulation<dim> &tria, const unsigned int n_refinements)
  {
    // according to the description in A FLEXIBLE, PARALLEL, ADAPTIVE
    // GEOMETRIC MULTIGRID METHOD FOR FEM (Clevenger, Heister, Kanschat,
    // Kronbichler): https://arxiv.org/pdf/1904.03317.pdf

    hyper_cube(tria, -1.0, +1.0);

    if (n_refinements == 0)
      return;

    for (int i = 0; i < static_cast<int>(n_refinements) - 3; i++)
      tria.refine_global();

    if (n_refinements >= 1)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            if (cell->center().norm() < 0.55)
              cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }

    if (n_refinements >= 2)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            if (0.3 <= cell->center().norm() && cell->center().norm() <= 0.43)
              cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }

    if (n_refinements >= 3)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            if (0.335 <= cell->center().norm() && cell->center().norm() <= 0.39)
              cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }

    // AssertDimension(tria.n_global_levels() - 1, n_refinements);
  }
} // namespace dealii::GridGenerator
