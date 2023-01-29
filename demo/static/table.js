$('.table').on('scroll', function () {
    $("#"+this.id+" > *").width($(this).width() + $(this).scrollLeft());
  }
);

$('#ref_q_lst').on(
	{ 
		"focus": function() {
    		this.selectedIndex = -1;
  	}, 
  	"change": function() {
      choice = $(this).val();
      this.blur();
      $('#input_question').val(choice)
    }
	}
);

function get_post_json() {
    query_info = {
        'question':$('#input_question').val()
    }
    return query_info
}

$('#query_cmd').click(function() {
    input_data = get_post_json()
    if (input_data['question'].trim() == '') {
        alert('Please input a question.')
        return
    }
    cursor_state = document.body.style.cursor
    btn_cursor_state = document.getElementById('query_cmd').style.cursor
    document.body.style.cursor = "progress"
    document.getElementById('query_cmd').style.cursor = "progress"
    $('#query_cmd').prop("disabled",true);

    request = $.ajax({
          type: "POST",
          contentType: "application/json; charset=utf-8",
          url: "/",
          data: JSON.stringify(input_data),
        });
    
    request.done(function (res_data) {
        $("#top_table_container").html(res_data)
        $(".top_table_btn").click(function(event) {
            export_table($(this).attr('table_rank'))
        });

        $('#query_cmd').prop("disabled",false);
        document.body.style.cursor = cursor_state
        document.getElementById('query_cmd').style.cursor = btn_cursor_state
    });

    request.fail(function (xhr, status, err){
        alert(status + '_' + err)
        $('#query_cmd').prop("disabled",false);
        document.body.style.cursor = cursor_state
        document.getElementById('query_cmd').style.cursor = btn_cursor_state
    });

});

function export_table(rank) {
    table_id = 'top_table_' + rank
    caption = $($('#' + table_id).find('caption')[0]).text()
    file_name = caption + '.csv'
    csv_data = get_table_csv(table_id)
    var blob = new Blob([csv_data], {type:'text/csv;charset=utf-8'})
    if (navigator.msSaveBlob) {
        navigator.msSaveBlob(blob, file_name)
    } else {
        var link = document.createElement("a")
        if (link.download != undefined) {
            var url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", file_name);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
}

function get_table_csv(table_id) {
    csv_out_lst = []
    col_name_lst = []
    col_ele_lst = $('#' + table_id).find('thead th')
    for (idx = 0; idx < col_ele_lst.length; idx++) {
        col_name = $(col_ele_lst[idx]).text()
        col_name_lst.push(csv_escape(col_name))
    }
    csv_title = col_name_lst.join(',')
    csv_out_lst.push(csv_title)

    row_eles = $('#' + table_id).find('tbody tr')
    for (row_idx = 0; row_idx < row_eles.length; row_idx++) {
        r_ele = $(row_eles[row_idx])
        td_eles = r_ele.children('td')

        row_value_lst = []
        for (col_idx = 0; col_idx < col_ele_lst.length; col_idx++) {
            cell_value = $(td_eles[col_idx]).text()
            row_value_lst.push(csv_escape(cell_value))
        }
        csv_row = row_value_lst.join(',')
        csv_out_lst.push(csv_row)
    }
    csv_data = csv_out_lst.join('\n')
    return csv_data 
}

function csv_escape(text) {
    text_out = '"' + text.replace(/"/g, '""') + '"'
    return text_out
}

