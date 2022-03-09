class TemplateTag:
    caption = '[c-a-p]'
    
    sub_name = '[s-u-b-n-a-m-e]'
    sub = '[s-u-b]'
    
    obj_name = '[o-b-j-n-a-m-e]'
    obj = '[o-b-j]'

    sub_none = '[sub-none]'
    obj_none = '[obj-none]'

    @staticmethod
    def get_annotated_text(caption, sub_name, sub, obj_name, obj):
        if sub is not None:
            out_text = (
                    f'{TemplateTag.caption} {caption} ' 
                    f'{TemplateTag.sub_name} {sub_name} {TemplateTag.sub} {sub} '
                    f'{TemplateTag.obj_name} {obj_name} {TemplateTag.obj} {obj}'
            )
        else:
            out_text = (
                    f'{TemplateTag.caption} {caption} ' 
                    f'{TemplateTag.obj_name} {obj_name} {TemplateTag.obj} {obj}'
            )

        return out_text
