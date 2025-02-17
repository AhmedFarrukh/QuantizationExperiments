:::{.cell}
### Transfer Models to Local Device (Optional)
Using the following instructions, you can use SSH to transfer the models (original and quantized) from the remote server to your own local device.
:::

:::{.cell}
Paste the following command into a local terminal. Note that `~/.ssh/id_rsa_chameleon` is the location of your SSH key, and `./experiment_models` is the directory on your local computer where these models will be stored. Feel free to change them as needed.
:::

:::{.cell .code}
```python
print(f'scp -ri ~/.ssh/id_rsa_chameleon cc@{reserved_fip}:/home/cc/models ./experiment_models')
```
:::
