# Q1 notes

## Task B

### Institution domain extraction
To extract the institution domain from each request, the hostname field was first split into its constituent components using the "." delimiter. For example, the Sheffield hostname, `www.shef.ac.uk`, is split into `["www", "shef", "ac", "uk"]`.

The institution domain is then defined as the final three labels of the hostname (e.g., `["shef", "ac", "uk"]`), which are then concatenated together (`shef.ac.uk`) to represent the institution domain. This way, different hosts from the same institution (e.g., `cs.shef.ac.uk`, `www.shef.ac.uk`) are grouped together (`shef.ac.uk`).


### Aggregation
After extracting the institution domain for each request, aggregation is performed by grouping the data by the institution domain and counting the number of occurrences in each group. As each row in the dataset represents a single request, this count corresponds directly to the total number of requests made by each institution.

The grouped DataFrame is then used to compare institutions based on their total request counts, enabling identification of those that generated more requests than the University of Sheffield.


## Task C

### Figure design choice
A bar chart is chosen to compare request counts across discrete company categories. It clearly shows the top 9 companies alongside the aggregated "All Other Companies", with different colours used to highlight the comparison and improve readability.


## Task E

### Observations