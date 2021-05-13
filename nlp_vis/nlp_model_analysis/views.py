from django.shortcuts import render
from nlp_model_analysis.models import nlp_info
from tensorflow import keras
from .eec_analysis import *
from django.http import JsonResponse
import plotly
# Create your views here.
live_models = {}
eec_df = init_eec()

"""
name: name of the model
description: any additional text information as description
path: path from site folder's root directory to the model
"""
def generate_model_info(name, description, path):
    nlp_inf = nlp_info(name=name,description=description, path=path)
    live_models[name] = init_predictor(path)
    return nlp_inf

def nlp_model_analysis(request):
    nlp_inf = generate_model_info('BERT model', 'BERT EEC benchmarks', 'my_predictor')
    context = {
        'nlp_info': nlp_inf,
    }
    return render(request, 'nlp_model_analysis.html', context)

def gender_bias_metrics(request):
    gender_res = gender_disparities(live_models['BERT model'], eec_df)
    gender_res.reverse()
    return JsonResponse(gender_res, safe=False)

def racial_bias_metrics(request):
    racial_res = racial_disparities(live_models['BERT model'], eec_df)
    racial_res.reverse()
    return JsonResponse(racial_res, safe=False)

def query_sentence(request):
    #fake_gender_res = [{"male": 1, "female" : 2}]
    # sentence = get_object_or_404(string, pk)
    # try:
    #     sentence = question.choice_set.get(pk=request.POST['choice'])
    # except (KeyError, Choice.DoesNotExist):
    #     # Redisplay the question voting form.
    #     return render(request, 'polls/detail.html', {
    #         'question': question,
    #         'error_message': "You didn't select a choice.",
    #     })
    # else:
    #     selected_choice.votes += 1
    #     selected_choice.save()
    #     # Always return an HttpResponseRedirect after successfully dealing
    #     # with POST data. This prevents data from being posted twice if a
    #     # user hits the Back button.
    #     return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
    if request.GET.get('q'):
        message = predict_sent(live_models['BERT model'], request.GET['q']).tolist()
        message = {"neg_sent_prob": message[0], "pos_sent_prob": message[1]}
    else:
        message = 'You submitted nothing!'
    print(message)
    return JsonResponse(json.dumps({"prediction": message}), safe=False)