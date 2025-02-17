:::{.cell}
## Release resources
:::

:::{.cell}
If you finish with your experimentation before your lease expires, release your resources and tear down your environment by running the following.

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
