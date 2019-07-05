import numpy as np


class SpatialHashTable(object):
    """
    Hash table used to split coordinate system into buckets. Objects can be assigned to all buckets
    that overlap with the provided volume. A hash table is useful in cases where it becomes
    inefficient to search through all items in the coordinate system. Assigning objects to buckets
    allows for a selected retrieval of neighboring objects.

    Written by: Nil Stolt Ansó, 05/07/2019
    """
    def __init__(self, dims, bucket_size):
        self.dims = dims
        self.n_dims = dims.shape[0]
        self.buckets_per_dim = np.ceil(dims / bucket_size).astype(np.int32)
        self.bucket_size = bucket_size  # Assuming buckets have equally sized sides
        self.n_buckets = int(np.prod(self.buckets_per_dim))
        self.buckets = {}
        self.clear_buckets()

    def get_nearby_objects(self, pos, radius):
        """
        Given a position and radius, retrieve all objects in the overlapping buckets.
        :param pos: Center of search volume.
        :param radius: Radius of search volume.
        :return: Objects in buckets overlapping with given volume.
        """
        cell_ids = self.get_ids_for_volume(pos, radius)
        return self.get_objects_from_buckets(cell_ids)

    def get_ids_for_volume(self, pos, radius):
        """
        Retrieve the IDs of all buckets overlapping with the volume with the given center position
        and given radius.
        :param pos: Center of search volume.
        :param radius: Radius of search volume.
        :return: IDs of buckets overlapping volume.
        """
        ids = set()
        lowest_pos = np.max((pos - radius, np.zeros((self.n_dims,))), axis=0)
        lowest_bucket_lower_bound = (lowest_pos - lowest_pos % self.bucket_size).astype(np.int32)
        highest_bucket_upper_bound = (np.min((self.dims, pos + radius + 1.0), axis=0)).astype(np.int32)

        for x in range(lowest_bucket_lower_bound[0], highest_bucket_upper_bound[0], self.bucket_size):
            for y in range(lowest_bucket_lower_bound[1], highest_bucket_upper_bound[1], self.bucket_size):
                for z in range(lowest_bucket_lower_bound[2], highest_bucket_upper_bound[2], self.bucket_size):
                    ids.add(self.get_id(x, y, z))
        return ids

    def get_id(self, x, y, z):
        """
        Get bucket ID containing the given Cartesian coordinate.
        :param x:
        :param y:
        :param z:
        :return:
        """
        return x // self.bucket_size + y // self.bucket_size * self.buckets_per_dim[0] + \
               z // self.bucket_size * self.buckets_per_dim[0] * self.buckets_per_dim[1]

    def get_objects_from_buckets(self, ids):
        """
        Given the IDs of buckets, return the union of every set obtained from each individual bucket.
        :param ids: Indices of buckets.
        :return: Union of objects found in those buckets.
        """
        objects = set()
        for i in ids:
            objects = objects.union(self.buckets[i])
        return objects

    def clear_buckets(self):
        """
        Remove all objects from all buckets in the hash table.
        :return:
        """
        for idx in range(self.n_buckets):
            self.buckets[idx] = set()

    def insert_object(self, obj, pos, radius):
        """
        Insert an object into all buckets that overlap with the volume with center 'pos' and
        radius 'radius'
        :param obj: Object to be inserted into buckets
        :param pos: Center of search volume.
        :param radius: Radius of search volume.
        :return:
        """
        idxs = self.get_ids_for_volume(pos, radius)
        for idx in idxs:
            self.buckets[idx].add(obj)

    def insert_objects(self, object_structure):
        """
        Insert a structure of objects into hash table.
        :param object_structure: Data structure where each row is of form (object, position, radius)
        :return:
        """
        for (obj, pos, radius) in object_structure:
            self.insert_object(obj, pos, radius)

    def get_dims(self):
        """
        Get dimensions of hash table in terms of the coordinate system.
        :return:
        """
        return self.dims

    def get_buckets_per_dim(self):
        """
        Get how many buckets lay in each dimension.
        :return: Tuple of number of buckets per dimension
        """
        return self.buckets_per_dim

    def get_buckets(self):
        """
        Get all buckets.
        :return:
        """
        return self.buckets

    def get_bucket_content(self, i):
        """
        Get all objects in bucket of the given ID.
        :param i: ID of the bucket
        :return: Objects in the bucket with given ID.
        """
        return self.buckets[i]

    def get_bucket_center(self, i):
        """
        Get the center coordinate (in terms of coordinate system) of bucket with given ID.
        :param i: Index of bucket.
        :return: Center coordinate of bucket.
        """
        center = np.empty((self.n_dims,))
        center[0] = i % self.dims[0] * self.bucket_size + self.bucket_size / 2
        for d in range(1, self.n_dims):
            center[d] = i // np.prod(self.dims[:d]) * self.bucket_size + self.bucket_size / 2
        return center

    def remove_object(self, obj, pos, radius):
        """
        Remove object from all buckets overlapping with volume with center 'pos' and
        radius 'radius'.
        :param obj: Object to be inserted into buckets
        :param pos: Center of search volume.
        :param radius: Radius of search volume.
        :return:
        """
        idxs = self.get_ids_for_volume(pos, radius)
        for idx in idxs:
            self.buckets[idx].remove(obj)
