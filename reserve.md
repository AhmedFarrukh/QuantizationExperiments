:::{.cell}
# Run a Single User Notebook Server on Chameleon
:::

:::{.cell}
This notebook describes how to run a single user Jupyter notebook server on Chameleon. This allows you to run experiments requiring bare metal access, storage, memory, GPU and compute resources on Chameleon using a Jupyter notebook interface. 
:::

:::{.cell}
## Provision the resource
:::

:::{.cell}
### Check resource availability
:::

:::{.cell}
This notebook will try to reserve a bare metal server on Chameleon - pending availability. You can see the hardware resources available here: [https://chameleoncloud.org/hardware/](https://chameleoncloud.org/hardware/).

Once you have selected the hardware resource, identify it's site, and then confirm availability using the following site-specific host calendars:  
- [TACC](https://chi.tacc.chameleoncloud.org/project/leases/calendar/host/)  
- [UC](https://chi.uc.chameleoncloud.org/project/leases/calendar/host/)  
- [NU](https://sl-ciab.northwestern.edu/project/leases/calendar/host/)  
- [NCAR](https://chi.hpc.ucar.edu/project/leases/calendar/host/)  
- [EVL](https://chi.evl.uic.edu/project/leases/calendar/host/)  
:::

:::{.cell}
### Chameleon configuration
:::

:::{.cell}
In the following cell, you can enter the site and node type.
:::

:::{.cell .code}
```python
import chi, os
chi.use_site("CHI@UC")
NODE_TYPE = "compute_cascadelake_r"
```
:::

:::{.cell}
You can also change your Chameleon project name (if not using the one that is automatically configured in the JupyterHub environment) in the following cell.
:::

:::{.cell .code}
```python
PROJECT_NAME = os.getenv('OS_PROJECT_NAME')
chi.set("project_name", PROJECT_NAME)
username = os.getenv('USER')
```
:::

If you need to change the details of the Chameleon server, e.g. use a different OS image, you can do that in the following cell.

:::{.cell .code}
```python
chi.set("image", "CC-Ubuntu20.04")
```
:::

:::{.cell}
### Reservation
:::

:::{.cell}
The following cell will create a reservation that begins now, and ends in 8 hours. You can modify the start and end date as needed.
:::

:::{.cell .code}
```python
from chi import lease


res = []
lease.add_node_reservation(res, node_type=NODE_TYPE, count=1)
lease.add_fip_reservation(res, count=1)
start_date, end_date = lease.lease_duration(days=0, hours=8)

l = lease.create_lease(f"{username}-{NODE_TYPE}", res, start_date=start_date, end_date=end_date)
l = lease.wait_for_active(l["id"])  #Comment this line if the lease starts in the future
```
:::

:::{.cell .code}
```python
# continue here, whether using a lease created just now or one created earlier
l = lease.get_lease(f"{username}-{NODE_TYPE}")
l['id']
```
:::

:::{.cell}
### Provisioning resources
:::

:::{.cell}
The following cell provisions resources. It will take approximately 10 minutes. You can check on its status in the Chameleon web-based UI, which can be accessed by selecting 'Instances' under the 'Compute' tab on the relevant site's webpage. For example, for a node on the CHI@UC site, you can use [https://chi.uc.chameleoncloud.org/project/instances/]( https://chi.uc.chameleoncloud.org/project/instances/). Come back here when it is in the RUNNING state.
:::


:::{.cell .code}
```python
from chi import server

reservation_id = lease.get_node_reservation(l["id"])
server.create_server(
    f"{username}-{NODE_TYPE}",
    reservation_id=reservation_id,
    image_name=chi.get("image")
)
server_id = server.get_server_id(f"{username}-{NODE_TYPE}")
server.wait_for_active(server_id)
```
:::

:::{.cell}
Associate an IP address with this server:
:::

:::{.cell .code}
```python
reserved_fip = lease.get_reserved_floating_ips(l["id"])[0]
server.associate_floating_ip(server_id,reserved_fip)
```
:::

:::{.cell}
and wait for it to come up:
:::

:::{.cell .code}
```python
server.wait_for_tcp(reserved_fip, port=22)
```
:::

:::{.cell}
## Install Required Libraries and Packages
:::

:::{.cell}
The following cells will install the libraries and packages required to initiate the Jupyter notebook server, and then execute the experiment notebook on this server.
:::

:::{.cell .code}
```python
from chi import ssh

node = ssh.Remote(reserved_fip)
```
:::

:::{.cell .code}
```python
node.run('sudo apt update')
node.run('sudo apt -y install python3-pip python3-dev')
node.run('sudo pip3 install --upgrade pip')
```
:::

:::{.cell}
### Install Python packages
:::

:::{.cell .code}
```python
node.run('python3 -m pip install --user matplotlib==3.7.5 gdown==5.2.0 pandas==2.0.3')
```
:::

:::{.cell}
### Set up Jupyter on server
:::
:::{.cell}
Install Jupyter:
:::

:::{.cell .code}
```python
node.run('python3 -m pip install --user  jupyter-core jupyter-client jupyter -U --force-reinstall')
```
:::

:::{.cell}
### Retrieve the materials
:::

:::{.cell}
Finally, get a copy of the notebooks that you will run:
:::

:::{.cell .code}
```python
node.run('git clone https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing.git')
```
:::

:::{.cell}
## Run a JupyterHub server
:::

:::{.cell}
Run the following cell:
:::

:::{.cell .code}
```python
print('ssh -L 127.0.0.1:8888:127.0.0.1:8888 cc@' + reserved_fip)
```
:::
then paste its output into a *local* terminal on your own device, to set up a tunnel to the Jupyter server. If your Chameleon key is not in the default location, you should also specify the path to your key as an argument, using `-i`. Leave this SSH session open.

Then, run the following cell, which will start a command that does not terminate:

:::{.cell .code}
```python
node.run("/home/cc/.local/bin/jupyter notebook --port=8888 --notebook-dir='DeepLearning-EdgeComputing/notebooks'")
```
:::

:::{.cell}
In the output of the cell above, look for a URL in this format:
```
http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Copy this URL and open it in a browser. Then, you can run `measuring_inference_times.ipynb` notebook that you'll see there.

If you need to stop and re-start your Jupyter server,

- Use Kernel > Interrupt Kernel *twice* to stop the cell above
- Then run the following cell to kill whatever may be left running in the background.
:::

:::{.cell .code}
```python
node.run("sudo killall jupyter-notebook")
```
:::

:::{.cell}
## Release resources
:::

:::{.cell}
If you finish with your experimentation before your lease expires, release your resources and tear down your environment by running the following (commented out to prevent accidental deletions).

This section is designed to work as a "standalone" portion - you can come back to this notebook, ignore the top part, and just run this section to delete your reasources. 
:::

:::{.cell}
Make sure to set the correct site first, by entering its name in the following cell.
:::

:::{.cell .code}
```python
import chi, os
from chi import lease, server
chi.use_site("CHI@UC")
```
:::

:::{.cell .code}
```python
# setup environment - if you made any changes in the top part, make the same changes here
import chi, os
from chi import lease, server

PROJECT_NAME = os.getenv('OS_PROJECT_NAME')
chi.set("project_name", PROJECT_NAME)


lease = chi.lease.get_lease(f"{username}-{NODE_TYPE}")
```
:::

:::{.cell .code}
```python
DELETE = False
# DELETE = True

if DELETE:
    # delete server
    server_id = chi.server.get_server_id(f"{username}-{NODE_TYPE}")
    chi.server.delete_server(server_id)

    # release floating IP
    reserved_fip =  chi.lease.get_reserved_floating_ips(lease["id"])[0]
    ip_info = chi.network.get_floating_ip(reserved_fip)
    chi.neutron().delete_floatingip(ip_info["id"])

    # delete lease
    chi.lease.delete_lease(lease["id"])

```
:::
