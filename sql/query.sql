select * from dataset d
join dataset_f1 df ON df.dataset_id =d.id
join f1 f1 ON f1.id=df.f1_id 
where d."name" ='regions_dataset';

select t.k, t."percent", d.version from dataset d
--select * from dataset d 
join dataset_topk dt on dt.dataset_id =d.id 
join topk t on t.id =dt.topk_id 
where d.name='br_dataset' and 
d.minimum =5 and 
d.width =512 and d.height =512 and 
d.color ='RGB' and 
t.k in (3, 5) and 
dt.classifier ='MLPClassifier' and 
d.model ='vgg16' and 
d.n_features =512 and 
t."rule"='sum';