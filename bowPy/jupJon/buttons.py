from ipywidgets import widgets,HBox,Layout,VBox


def button_gen(label,button_func,button_inputs = [],display_now = True):
    box_layout = Layout(
                display='flex',
                flex_flow='row',
                align_items='flex-start',
                # border='solid',
                # width='50%'
                )

    button = widgets.Button(description=label,layout =box_layout)
    output = widgets.Output()



    if display_now:
        display(button, output)
    button.f_out = []

    def on_button_clicked(b):
        with output:
            button.f_out.append(button_func(*button_inputs))

    button.on_click(on_button_clicked)
    return(button)


def gen_gridbox(lst,n = 2):
    bxs =[HBox(lst[i:i + n]) for i in range(0, len(lst), n)]
    return(VBox(children=bxs))