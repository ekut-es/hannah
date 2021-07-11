for model in tc-res20 tc-res6 tc-res8 tc-res18; do
    hannah-train model=$model experiment_id=train_teachers scheduler=1cycle optimizer=sgd module.num_workers=8 scheduler.max_lr=1.2 trainer.max_epochs=30
    cp -r trained_models/$model/best.ckpt teachers/hannah/$model.ckpt
done
