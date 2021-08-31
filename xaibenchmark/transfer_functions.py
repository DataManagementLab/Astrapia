def generate_default_transfer_functions(add_transfer):

    def area_norm(area, get_explained_instance):
        return area() ** (1 / len(get_explained_instance()))
    add_transfer(area_norm)

    def furthest_distance(distance, get_explained_instance, get_neighborhood_instance):
        return max(0, 0, *[distance(get_explained_instance(), i) for i in get_neighborhood_instance()])
    add_transfer(furthest_distance)

    pass
