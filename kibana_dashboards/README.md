## DRL Train Dashboard ##

TODO: explain about the features and sections


### For Developers ###
To ensure that some results are not affected by the Training ID control, we should create a copy of the `drl_training` dataview and use the two dataviews for different purposes.

In the original `drl_training` dataview, we add a dynamic field that copies the value of `training_id`. The Training ID control will then filter logs based on this dynamic field.

As a result:

* Tables that should be affected by the *Training ID control* should use the original `drl_training` dataview, which contains the dynamic field used for filtering.
* Tables that should not be affected by the *Training ID control* (for example, the Baseline and "All Execution Results" tables) should use the copied `drl_training` dataview. Since this copy does not contain the dynamic field used by the Training ID control, its logs will not be filtered by that control.

This approach allows us to apply the Training ID filter selectively.
