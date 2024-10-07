const ctx = document.getElementById("chart");
window.onload = () => {
    ctx.classList.add("visible");
}

const button = document.getElementById("toggle");

function toggle_graph() {
  if (ctx.style.display === "none") {
      button.innerText = "Hide Graph";
    ctx.style.display = "block";
  } else {
      button.innerText = "Show Graph";
    ctx.style.display = "none";
  }

}

function chart_data(current) {
  return {
    labels: current.t,
    datasets: [
      {
        label: "Probability of quake",
        data: current.p,
        fill: false,
        borderColor: "#5c5b5b",
        tension: 0.1,
      },
    ],
  };
}

function chart_options(current) {
  return {
    plugins: {
      title: {
        display: true,
        text: current.name + " Predicted Quake Probability",
      },
      legend: {
        display: false,
      },
    },
  };
}

let chart = new Chart(ctx, {
  type: "line",
  data: chart_data(data[0]),
  options: chart_options(data[0]),
});

let data_i = 0;

function prev() {
  data_i = data_i - 1;

  if (data_i < 0) {
    data_i = 7;
  }

  console.log(data_i);
  let current = data[data_i];

  chart.data.datasets[0].data = current.p;
  (chart.options.plugins.title.text =
    current.name + " Predicted Quake Probability"),
    chart.update();
}

function next() {
  data_i = (data_i + 1) % 8;

  if (data_i >= 8) {
    data_i = 0;
  }

  console.log(data_i);

  let current = data[data_i];

  chart.data.datasets[0].data = current.p;
  (chart.options.plugins.title.text =
    current.name + " Predicted Quake Probability"),
    chart.update();
}
