
namespace dealii::parallel
{
  template <int dim, int spacedim = dim>
  std::function<
    unsigned int(const typename Triangulation<dim, spacedim>::cell_iterator &,
                 const typename Triangulation<dim, spacedim>::CellStatus)>
  hanging_nodes_weighting(const double weight)
  {
    return [weight](const auto &cell, const auto &) -> unsigned int {
      bool flag = false;

      for (const auto f : cell->face_indices())
        if (!cell->at_boundary(f) &&
            (cell->neighbor(f)->has_children() ||
             cell->level() != cell->neighbor(f)->level()))
          flag = true;

      if (flag)
        return 10000 * weight;
      else
        return 10000;
    };
  }


} // namespace dealii::parallel