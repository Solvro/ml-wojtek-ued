import equinox as eqx
import orbax.checkpoint as ocp


def save_checkpoint(ckptr, step, model, opt_state):
    array_model, _ = eqx.partition(model, eqx.is_array)
    ckptr.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(array_model),
            opt_state=ocp.args.StandardSave(opt_state),
        ),
    )


def restore_checkpoint(ckptr, step, template_model, template_opt_state):
    array_template, static = eqx.partition(template_model, eqx.is_array)
    restored = ckptr.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(array_template),
            opt_state=ocp.args.StandardRestore(template_opt_state),
        ),
    )
    model = eqx.combine(restored.model, static)
    return model, restored.opt_state
