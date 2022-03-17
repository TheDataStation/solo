class TemplateTag:
    title = '[T]'
    
    sub_name = '[SC]'
    sub = '[S]'
    
    obj_name = '[OC]'
    obj = '[O]'

    sub_none = ''
    obj_none = ''

    @staticmethod
    def get_annotated_text(title, sub_name, sub, obj_name, obj):
        assert (title is not None)
        if sub_name is None:
            sub_name = ''
        if sub is None:
            sub = ''
        assert (obj_name is not None)
        assert(obj is not None)
        out_text = (
                f'{TemplateTag.title} {title} ' 
                f'{TemplateTag.sub_name} {sub_name} {TemplateTag.sub} {sub} '
                f'{TemplateTag.obj_name} {obj_name} {TemplateTag.obj} {obj}'
        )
        return out_text
