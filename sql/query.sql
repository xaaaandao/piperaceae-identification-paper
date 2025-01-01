select df.classifier, f."rule" , f.mean , f.std , d.model
from dataset d , dataset_f1 df, f1 f
where
    df.dataset_id =d.id and
    df.f1_id =f.id and
    d.height = 512 and d.width =512 and
    d.region = 'South' and
    f."rule" = 'sum' and
    d.color ='RGB' and
    d.model ='vit_large' and
    d.minimum =10;

select dt.classifier, t."rule", t.mean, t.k , t.std , d.model, t."percent"
from dataset d , dataset_topk dt , topk t
where
    dt.dataset_id =d.id and
    dt.topk_id = t.id and
    d.height = 512 and d.width =512 and
    d.region = 'South' and
    t."rule" = 'sum' and
    d.color ='RGB' and
    d.model ='vit_large' and
    (t.k=3 or t.k=5) and
    d.minimum =10;

select *
from dataset d
where d."name"='pr_dataset';

select *
from dataset d
where d."name"='br_dataset';

select *
from dataset d
where d."name"='regions_dataset' and
    d.region ='South';