"""
>>> hydra.initialize(config_path='met/conf', version_base="1.3")
>>> cfg = hydra.compose(config_name='config')
"""
import hydra
import omegaconf

import met.callbacks
import met.constants
import met.data
import met.eval

constants = met.constants.Constants()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    train_dl = hydra.utils.instantiate(cfg.data.train)
    optim = hydra.utils.instantiate(cfg.model.optimizer)
    try:
        scheduler = hydra.utils.instantiate(cfg.model.scheduler)
    except omegaconf.errors.ConfigAttributeError:
        scheduler = None
    model = hydra.utils.instantiate(cfg.model.nn, optim=optim, scheduler=scheduler)
    callbacks = met.callbacks.instantiate_callbacks(cfg.callbacks)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_dl)
    trainer.checkpoint_callback.to_yaml()

    try:
        meta = {"model": cfg.model.name}
        model = met.eval.Model(model, trainer.checkpoint_callback.best_model_path)
        cls = hydra.utils.instantiate(cfg.eval.cls)
        x_train, y_train, x_test, y_test = met.eval.preprocess_data(
            model,
            train_data=hydra.utils.instantiate(cfg.eval.train_data),
            test_data=hydra.utils.instantiate(cfg.eval.test_data),
        )
        results = met.eval.evaluate_classifier(model, cls, x_train, y_train, x_test, y_test)
        meta.update(results)
        met.eval.to_json(results=meta, filepath=constants.OUTPUTS.joinpath("results.json"))
    except NotImplementedError:
        print("No encoder method implemented. Cannot evaluate linear discrimination.")


if __name__ == "__main__":
    main()
