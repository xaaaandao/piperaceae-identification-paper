select * from dataset d
join dataset_f1 df ON df.dataset_id =d.id
join f1 f1 ON f1.id=df.f1_id 
where d."name" ='regions_dataset';

