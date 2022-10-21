class TemplateTag:
    para_tag = 'P-A-R'
    Para_Start = '[%s]' % para_tag
    Para_End = '[/%s]' % para_tag

    triple_tag = 'T-R-P'
    Triple_Start = '[%s]' % triple_tag
    Triple_End = '[/%s]' % triple_tag

    subject_tag = 'S-U-B'
    Subject_Start = '[%s]' % subject_tag
    Subject_End = '[/%s]' % subject_tag

    rel_tag = 'R-E-L'
    Rel_Start = '[%s]' % rel_tag
    Rel_End = '[/%s]' % rel_tag

    object_tag = 'O-B-J'
    Object_Start = '[%s]' % object_tag
    Object_End = '[/%s]' % object_tag

    Meta_Tags = [Para_Start, Para_End, Rel_Start, Rel_End, Triple_Start, Triple_End]

    @staticmethod
    def get_annotated_triple(sub, rel, obj):
        out_triple = (
                f'{TemplateTag.Para_Start} ' 
                f'{TemplateTag.Triple_Start} '
                f'{TemplateTag.Subject_Start} {sub} {TemplateTag.Subject_End} '
                f'{TemplateTag.Rel_Start} {rel} {TemplateTag.Rel_End} '
                f'{TemplateTag.Object_Start} {obj} {TemplateTag.Object_End} '
                f'{TemplateTag.Triple_End} '
                f'{TemplateTag.Para_End}'
        )
        return out_triple
