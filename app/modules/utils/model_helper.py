from sqlalchemy.orm import class_mapper
from sqlalchemy.orm.exc import DetachedInstanceError

class ModelHelper:
    @staticmethod
    def model_to_dict(model, max_depth=1, current_depth=0):
        if model is None or current_depth > max_depth:
            return None

        result = {}

        try:
            mapper = class_mapper(model.__class__)
        except:
            # If it's not a SQLAlchemy model class, return the object as is
            return model

        for key in mapper.column_attrs.keys():
            result[key] = getattr(model, key)

        # Handle relationships
        for rel_name, rel_attr in mapper.relationships.items():
            try:
                related_obj = getattr(model, rel_name)
                if related_obj is not None:
                    if isinstance(related_obj, list):
                        result[rel_name] = [ModelHelper.model_to_dict(item, max_depth, current_depth + 1) for item in related_obj if item is not None]
                    else:
                        result[rel_name] = ModelHelper.model_to_dict(related_obj, max_depth, current_depth + 1)
            except DetachedInstanceError:
                # Skip this relationship if it's not loaded
                pass

        return result