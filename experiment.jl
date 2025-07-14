using CSV
using DataFrames
using Plots


df = DataFrame(CSV.File(download("http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt"),
                   header=["year", "hare", "lynx"],
                   delim=' ',
                   ignorerepeated=true))


plot(df.year, [df.hare df.lynx], label=["Hare" "Lynx"], xlabel="Year", ylabel="Population", lw=2, title="Hare and Lynx Populations Over Time")

