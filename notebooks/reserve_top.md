:::{.cell}
## Setup a Resource on Chameleon

The following steps will allow you to reserve and bring up a resource running on Chameleon's bare metal servers. 

You can see the hardware resources available here: [https://chameleoncloud.org/hardware/](https://chameleoncloud.org/hardware/).

Once you have selected the hardware resource, identify it's site, and then confirm availability using the following site-specific host calendars:  
- [TACC](https://chi.tacc.chameleoncloud.org/project/leases/calendar/host/)  
- [UC](https://chi.uc.chameleoncloud.org/project/leases/calendar/host/)  
- [NU](https://sl-ciab.northwestern.edu/project/leases/calendar/host/)  
- [NCAR](https://chi.hpc.ucar.edu/project/leases/calendar/host/)  
- [EVL](https://chi.evl.uic.edu/project/leases/calendar/host/)  
:::


:::{.cell}
### Chameleon Configuration
In the following cell, you can enter the site and node type.
:::

:::{.cell .code}
```python
import chi, os
SITE_NAME = "CHI@UC"
chi.use_site(SITE_NAME)
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

:::{.cell}
If you need to change the details of the Chameleon server, e.g. use a different OS image, you can do that in the following cell.
:::

:::{.cell .code}
```python
chi.set("image", "CC-Ubuntu20.04")
```
:::

:::{.cell}
### Reservation
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
### Install Required Libraries and Packages
:::

:::{.cell}
The following cells will install the neccessary packages.
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
