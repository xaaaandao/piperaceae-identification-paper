select * from dataset d
                  join dataset_f1 df ON df.dataset_id =d.id
                  join f1 f1 ON f1.id=df.f1_id
where d."name" ='regions_dataset';

select f1.mean as f1, d.minimum, d.n_samples, d.width, d.height, d.model, d."version"  from dataset d
--select * from dataset d
                                                                                                join dataset_f1 df ON df.dataset_id =d.id
                                                                                                join f1 f1 ON f1.id=df.f1_id
where d."name" ='br_dataset' and
    df.classifier='MLPClassifier' and
    d.width=512 and
    d.minimum=20 and
    d.model='vgg16' and
    d.color ='RGB' and
    d.n_features =512;

select * from topk t ;


select t.k, t.percent, d.minimum, d.n_samples, d.width, d.height, d.model, d."version"  from dataset d
--select * from dataset d
                                                                                                 join dataset_topk dt ON dt.dataset_id =d.id
                                                                                                 join topk t ON t.id=dt.topk_id
where d."name" ='br_dataset' and
    dt.classifier='MLPClassifier' and
    d.width=512 and
    d.minimum=20 and
    d.model='vgg16' and
    d.color ='RGB' and
    d.n_features =512 and
    t.k in (3, 5);